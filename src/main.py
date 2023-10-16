"""Analysis of unsupervised wisdom data."""
import datetime
import json
import os
import pickle
import re
import time
from collections import Counter
from typing import Any

import hdbscan
import nltk
import numpy as np
import openai
import pandas as pd
import plotly.graph_objects as go
import spacy
import torch
import umap
import yake
from hdbscan.hdbscan_ import HDBSCAN
from hyperopt import fmin, hp, partial, space_eval, STATUS_OK, tpe, Trials
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Figure
from scipy.spatial.distance import cosine
from tqdm import tqdm, trange
from transformers import (
    AutoTokenizer,
    BertLMHeadModel,
    BertModel,
    BertTokenizer,
    MvpForConditionalGeneration,
    MvpTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

medical_terms = {
    "&": "and",
    "***": "",
    ">>": "clinical diagnosis",
    "@": "at",
    "abd": "abdomen",
    "af": "accidental fall",
    "afib": "atrial fibrillation",
    "aki": "acute kidney injury",
    "am": "morning",
    "ams": "altered mental status",
    "bac": "blood alcohol content",
    "bal": "blood alcohol level,",
    "biba": "brought in by ambulance",
    "c/o": "complains of",
    "chi": "closed-head injury",
    "clsd": "closed",
    "cpk": "creatine phosphokinase",
    "cva": "cerebral vascular accident",
    "dx": "clinical diagnosis",
    "ecf": "extended-care facility",
    "er": "emergency room",
    "etoh": "ethyl alcohol",
    "eval": "evaluation",
    "fd": "fall detected",
    "fx": "fracture",
    "fxs": "fractures",
    "glf": "ground level fall",
    "h/o": "history of",
    "htn": "hypertension",
    "hx": "history of",
    "inj": "injury",
    "inr": "international normalized ratio",
    "intox": "intoxication",
    "l": "left",
    "loc": "loss of consciousness",
    "lt": "left",
    "mech": "mechanical",
    "mult": "multiple",
    "n.h.": "nursing home",
    "nh": "nursing home",
    "p/w": "presents with",
    "pm": "afternoon",
    "pt": "patient",
    "pta": "prior to arrival",
    "pts": "patient's",
    "px": "physical examination",
    "r": "right",
    "r/o": "rules out",
    "rt": "right",
    "s'd&f": "slipped and fell",
    "s/p": "after",
    "sah": "subarachnoid hemorrhage",
    "sdh": "acute subdural hematoma",
    "sts": "sit-to-stand",
    "t'd&f": "tripped and fell",
    "tr": "trauma",
    "uti": "urinary tract infection",
    "w/": "with",
    "w/o": "without",
    "wks": "weeks",
}


def clean_narrative(text: str) -> str:
    """
    Preprocess the input text to provide a cleaner narrative.

    Inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/50/
    This function performs several text preprocessing steps to transform medical narratives into a more readable format:
    1. Converts the text to lowercase.
    2. Modifies mention of 'DX' for readability.
    3. Removes specific age and sex identifications, replacing them with the general term "patient".
    4. Translates certain medical terms and symbols into their user-friendly counterparts.
    5. Formats the text into user-friendly sentences, ensuring proper capitalization and spacing.

    Parameters
    ----------
    text : str
        The raw medical narrative text to be cleaned.

    Returns
    -------
    str
        The preprocessed text in a more user-friendly format.

    """
    # lowercase everything
    text = text.lower()

    # unglued DX
    regex_dx = r"([ˆ\W]*(dx)[ˆ\W]*)"
    text = re.sub(regex_dx, r". dx: ", text)

    # remove age and sex identifications
    # regex to capture age and sex (not perfect but captures almost all the cases)
    regex_age_sex = r"(\d+)\s*?(yof|yf|yo\s*female|yo\s*f|yom|ym|yo\s*male|yo\s*m)"
    if age_sex_match := re.search(regex_age_sex, text):
        # age = age_sex_match[1]
        sex = age_sex_match[2]
        if "f" in sex or "m" in sex:
            # text = text.replace(age_sex_match.group(0), f"{age} years old female")
            text = text.replace(age_sex_match[0], "patient")
    # translate medical terms
    for term, replacement in medical_terms.items():
        if term in ["@", ">>", "&", "***"]:
            pattern = rf"({re.escape(term)})"
            text = re.sub(pattern, f" {replacement} ", text)  # force spaces around replacement

        else:
            pattern = rf"(?<!-)\b({re.escape(term)})\b(?!-)"
            text = re.sub(pattern, replacement, text)

    # user-friendly format
    sentences = sent_tokenizer.tokenize(text)
    sentences = [sent.capitalize() for sent in sentences]
    sent = " ".join(sentences)
    for char in ["+", "-", "?"]:
        sent = sent.replace(char, " ")
    return sent


def fill_variables(df: pd.DataFrame, vm: pd.DataFrame) -> pd.DataFrame:
    """
    Replace values in a DataFrame based on a mapping provided by another DataFrame.

    This function iterates through each column in the primary DataFrame (`df`) and checks for the presence of that
    column in the mapping DataFrame (`vm`). If the column is found in the mapping DataFrame, the values in the primary
    dataframe are replaced based on the mapping. If no mapping exists for a specific value, the original value remains
    unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        The primary DataFrame containing the data whose values may need to be replaced.
    vm : pd.DataFrame
        The mapping DataFrame. Each column name should match a potential column name in `df`, and the values in the
        columns should correspond to the mapping from old values to new values.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with values replaced based on the mapping provided by `vm`.

    """
    final_dict = {}
    for col in df.columns:
        if col in list(vm):
            temp_list = []
            for j in df[col]:
                try:
                    mapped_value = vm[col][str(j)]
                    temp_list.append(mapped_value)
                except Exception:
                    temp_list.append(j)
            final_dict[col] = temp_list
        else:
            final_dict[col] = df[col].values

    return pd.DataFrame(final_dict)


def create_untext(df_final: pd.DataFrame) -> None:
    """
    Process the 'text' column in a DataFrame, removing specific words and applying various NLP transformations.

    The function performs several preprocessing tasks on the 'text' column of the input DataFrame:
    1. Extracts unique values from the 'race' column.
    2. Gathers synonyms for the words 'fall' and 'patient'.
    3. Processes each row in the DataFrame, applying:
        - Value extraction for multiple columns.
        - Contractions expansion.
        - Removal of stopwords, specified strings, and synonyms.
        - Tokenization, stemming, and lemmatization.
    4. After processing, the cleaned text is stored in a new column 'untext' in the original DataFrame.

    Parameters
    ----------
    df_final : pd.DataFrame
        Input DataFrame containing the 'text' column to be processed and other columns used for the extraction
        of strings to be removed from the text.

    Returns
    -------
    None
        The function modifies the input DataFrame in-place, adding the 'untext' column.

    """
    diag, num = np.unique(df_final["race"], return_counts=True)
    pd.DataFrame({"Column": diag, "Data Points": num})

    synonyms_fall = {lemma.name().lower() for syn in wn.synsets("fall") for lemma in syn.lemmas()}
    synonyms_fall.update(["fell", "falling", "fallen", "clinical", "diagnosis", "onto", "closed"])
    synonyms_patient = {lemma.name().lower() for syn in wn.synsets("patient") for lemma in syn.lemmas()}
    synonyms = synonyms_fall.union(synonyms_patient)

    nlp = spacy.load("en_core_web_sm")

    untext = []
    for i in trange(df_final.shape[0]):
        words = []
        for col in [
            "age",
            "sex",
            "race",
            "other_race",
            "hispanic",
            "diagnosis",
            "other_diagnosis",
            "diagnosis_2",
            "other_diagnosis_2",
            "body_part",
            "body_part_2",
            "disposition",
            "location",
            "fire_involvement",
            "alcohol",
            "drug",
            "product_1",
            "product_2",
            "product_3",
        ]:
            val = str(df_final[col].iloc[i])
            if val != "nan":
                words.append(val)

        text = df_final["text"].iloc[i]
        strings_to_remove = words

        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        text = text.lower().replace(r"/'t* ", " not ")

        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "shan't": "shall not",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            # Add more contractions if needed
        }

        # Create a regex pattern for finding the contractions in the text
        pattern = re.compile(r"\b(" + "|".join(contractions.keys()) + r")\b")

        # Replace the contraction with its expanded form
        text = pattern.sub(lambda x: contractions[x.group(0)], text)
        # print(text)

        # stemmed_text_words = [ps.stem(w) for w in word_tokenize(text)]
        # print(stemmed_text_words)
        stemmed_strings_to_remove = {ps.stem(w) for s in strings_to_remove for w in word_tokenize(s.lower())}
        result_words = [
            word
            for word in word_tokenize(text)
            if ps.stem(word.lower()) not in stemmed_strings_to_remove
            and word.lower() not in stop_words
            and word.lower() not in synonyms
        ]
        result_text = " ".join(result_words)
        for char in [".", ":"]:
            result_text = result_text.replace(char, " ")

        result_text = re.sub(r"\s+", " ", result_text)
        # print(result_text)
        # print()
        doc = nlp(result_text)
        rkw = ""
        for token in doc:
            rkw = rkw + token.lemma_ + " "
        untext.append(result_text)

    df_final["untext"] = untext


def create_embedding(x: list[str], filename: str = "untext_embed_final", step: int = 100) -> np.ndarray:
    """
    Generate embeddings for a list of text using the OpenAI's embedding model.

    If embeddings for the input list have already been saved with the given filename, they will be loaded from the file
    instead of recomputing them.

    Parameters
    ----------
    x : list[str]
        List of text strings for which embeddings are to be generated.

    filename : str, default="untext_embed_final"
        The filename (without extension) under which embeddings are saved/loaded as a pickle file.

    step : int, default=100
        The batch size to use when making calls to the OpenAI embedding model. Helps in sending batches of text for
        embedding generation rather than one by one.

    Returns
    -------
    embed : np.ndarray
        A numpy array containing the embeddings for each text string in 'x'.

    Notes
    -----
    The function uses OpenAI's `text-embedding-ada-002` model to generate the embeddings. If making frequent requests,
    remember that there may be rate limits, hence the function incorporates a sleep time between successive calls.

    """
    if os.path.exists(f"{filename}.pkl"):
        with open(f"{filename}.pkl", "rb") as f:
            embed = pickle.load(f)
    else:
        size = len(x)
        embeds = []
        for i in trange(0, size, step):
            k = size if i + step - 1 > size else i + step
            response = openai.Embedding.create(input=x[i:k], model="text-embedding-ada-002")["data"]
            for j in range(len(response)):
                ed = response[j]["embedding"]
                embeds.append(ed)
            time.sleep(1)

        embed = np.array(embeds)
        print(embed.shape)
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(embed, f)

    return embed


def generate_clusters(
    message_embeddings: np.ndarray, n_neighbors: int, n_components: int, min_cluster_size: int, random_state: int = None
) -> tuple[HDBSCAN, np.ndarray]:
    """
    Generate clusters from the provided message embeddings using UMAP for dimensionality reduction and HDBSCAN.

    Parameters
    ----------
    message_embeddings : np.ndarray
        Array containing the embeddings for each message.
    n_neighbors : int
        Number of neighboring points used in local neighborhood for UMAP.
    n_components : int
        Number of dimensions to which the data should be reduced using UMAP.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN. Smaller clusters are treated as noise.
    random_state : int, optional
        The seed used by the random number generator for UMAP. Defaults to None.

    Returns
    -------
    tuple[HDBSCAN, np.ndarray]
        A tuple containing:
        1. An HDBSCAN object fitted with the UMAP reduced embeddings.
        2. The reduced message embeddings after applying UMAP.

    """
    umap_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=n_components, metric="cosine", random_state=random_state
    ).fit_transform(message_embeddings)

    return (
        hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom").fit(
            umap_embeddings
        ),
        umap_embeddings,
    )


def score_clusters(clusters: HDBSCAN, prob_threshold: float = 0.05) -> tuple[int, float]:
    """
    Evaluate the given clusters by calculating the number of unique labels and the cost based on probabilities.

    Parameters
    ----------
    clusters : HDBSCAN
        The HDBSCAN object after fitting data, containing labels and membership probabilities for each data point.
    prob_threshold : float, optional
        The threshold below which a data point's cluster membership probability is considered uncertain or noisy.

    Returns
    -------
    tuple[int, float]
        A tuple containing:
        1. The number of unique clusters (label_count).
        2. The cost, represented as the fraction of data points with probabilities below the given threshold.

    """
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num

    return label_count, cost


def objective(params: dict, embeddings: np.ndarray, label_lower: int, label_upper: int) -> dict:
    """
    Objective function for hyperparameter optimization with hyperopt.

    Function inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/52/
    Evaluates the clustering results given specific hyperparameters, returning the associated loss.
    The loss incorporates constraints on the desired number of clusters.

    Parameters
    ----------
    params : dict
        Dictionary containing the hyperparameters to be optimized, including:
        - n_neighbors: Number of neighboring points for UMAP.
        - n_components: Number of dimensions for UMAP dimensionality reduction.
        - min_cluster_size: Minimum cluster size for HDBSCAN.
        - random_state: Seed for reproducibility in UMAP.
    embeddings : np.ndarray
        Array of embeddings for which clustering should be performed.
    label_lower : int
        Minimum desired number of unique cluster labels.
    label_upper : int
        Maximum desired number of unique cluster labels.

    Returns
    -------
    dict
        A dictionary containing:
        - loss: A value representing the optimization target, lower is better.
        - label_count: The number of unique clusters generated.
        - status: Status of the optimization, typically STATUS_OK for hyperopt.

    """
    clusters, _ = generate_clusters(
        embeddings,
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        min_cluster_size=params["min_cluster_size"],
        random_state=params["random_state"],
    )

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)
    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty
    return {"loss": loss, "label_count": label_count, "status": STATUS_OK}


def bayesian_search(
    embeddings: np.ndarray, space: dict, label_lower: int, label_upper: int, max_evals: int = 100
) -> tuple[Any, HDBSCAN, Trials]:
    """
    Perform Bayesian optimization using hyperopt to search for optimal hyperparameters for clustering.

    Function inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/52/
    This function searches the hyperparameter space to find the best set of parameters that minimizes the objective
    function defined for clustering. The objective function aims to find optimal clustering parameters based on
    certain constraints such as desired number of clusters.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of embeddings for which clustering should be performed.
    space : dict
        Hyperparameter space for hyperopt, defining the range and distribution for each hyperparameter.
    label_lower : int
        Minimum desired number of unique cluster labels.
    label_upper : int
        Maximum desired number of unique cluster labels.
    max_evals : int, optional
        Maximum number of evaluations during the Bayesian optimization. Defaults to 100.

    Returns
    -------
    tuple[Any, HDBSCAN, Trials]
        A tuple containing:
        - best_params: Dictionary of best hyperparameters found.
        - best_clusters: HDBSCAN object fitted with the embeddings using the best hyperparameters.
        - trials: hyperopt Trials object containing details of all the evaluations.

    """
    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params = space_eval(space, best)
    print("best:")
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters, _ = generate_clusters(
        embeddings,
        n_neighbors=best_params["n_neighbors"],
        n_components=best_params["n_components"],
        min_cluster_size=best_params["min_cluster_size"],
        random_state=best_params["random_state"],
    )

    return best_params, best_clusters, trials


def search_param(embed: np.ndarray) -> tuple[Any, object, Trials]:
    """
    Perform bayesian search on hyperopt hyperparameter space to find the best parameters for clustering.

    Function inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/52/
    The function searches for optimal hyperparameters for clustering using the HDBSCAN algorithm. It uses the hyperopt
    library to conduct a Bayesian search over a predefined hyperparameter space.

    Parameters
    ----------
    embed : np.ndarray
        Embeddings for which clustering parameters need to be optimized.

    Returns
    -------
    best_param_use : Any
        Dictionary containing the best parameters found during the search.

    best_clusters_use : object
        HDBSCAN clustering object resulting from using the best parameters on the provided embeddings.

    trials_use : Trials
        Trials object from hyperopt containing information about all the trials conducted during the search.

    """
    hspace = {
        "n_neighbors": hp.choice("n_neighbors", range(3, 25)),
        "n_components": hp.choice("n_components", [3]),
        "min_cluster_size": hp.choice("min_cluster_size", [50, 100, 150]),
        "random_state": 33,
    }

    label_lower = 30
    label_upper = 100
    max_evals = 100

    best_param_use, best_clusters_use, trials_use = bayesian_search(
        embed, space=hspace, label_lower=label_lower, label_upper=label_upper, max_evals=max_evals
    )

    return best_param_use, best_clusters_use, trials_use


def create_plot_3d(umap_embeddings: np.ndarray, clusters: HDBSCAN, label_count: int, flag: bool, title: str) -> Figure:
    """
    Create a 3D scatter plot visualizing the clusters from HDBSCAN in an embedded space.

    This function takes UMAP embeddings and their corresponding HDBSCAN cluster assignments to visualize the data
    points in a 3D scatter plot. Each cluster will be represented by a unique color, and noise (unassigned data points)
    will have its own designation.

    Parameters
    ----------
    umap_embeddings : np.ndarray
        UMAP embeddings of the data in a 3-dimensional space.

    clusters : HDBSCAN
        HDBSCAN clustering object containing cluster assignments for each data point in the embeddings.

    label_count : int
        Total number of clusters excluding noise.

    flag : bool
        If True, includes noise (data points not assigned to any cluster) in the visualization.
        If False, only clusters are visualized.

    title : str
        Title of the plot.

    Returns
    -------
    fig : Figure
        A 3D scatter plot visualizing the clusters in the embedded space.

    """
    cluster_range = range(-1, label_count) if flag else range(label_count)
    fig = go.Figure()
    for cluster in cluster_range:
        cluster_data = umap_embeddings[clusters.labels_ == cluster]

        # Providing a name to the cluster or marking it as noise
        name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        # Adding the cluster data to the plotly figure
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data[:, 0],
                y=cluster_data[:, 1],
                z=cluster_data[:, 2],
                mode="markers",
                marker=dict(size=3),
                name=name,
            )
        )

    # Setting the layout of the 3D plot
    fig.update_layout(
        title={"text": f"{title}", "y": 0.95, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig


def gen_keywords(text: str, num_keywords: int = 100) -> list[tuple[str, float]]:
    """
    Extract keywords from a given text using the YAKE (Yet Another Keyword Extractor) algorithm.

    Function inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/11/
    This function is designed to take a text input and return a specified number of keywords.
    It employs the YAKE algorithm, which is unsupervised and language-independent, to detect key phrases in the text
    based on their statistical, positional, and syntactic features.

    Parameters
    ----------
    text : str
        The input text from which keywords will be extracted.

    num_keywords : int, optional (default=100)
        The number of keywords to be returned from the extraction process.

    Returns
    -------
    list[tuple[str, float]]
        A list of tuples where each tuple contains a keyword and its corresponding YAKE score. The list is ordered by
        the score, with the most relevant keywords appearing first.

    """
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    deduplication_algo = "seqm"
    window_size = 1

    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        dedupFunc=deduplication_algo,
        windowsSize=window_size,
        top=num_keywords,
        features=None,
    )
    return custom_kw_extractor.extract_keywords(text)


def create_summary(documents: str, model: str = "gpt-4") -> str:
    """
    Generate a concise summary of the provided documents using the specified model, defaulting to "gpt-4".

    The function uses OpenAI's ChatCompletion API to create a summary based on the input documents.
    By providing the documents and specifying a model (e.g., "gpt-4"), the function instructs the model
    to condense the information and return a summary.

    Parameters
    ----------
    documents : str
        The input text or content that needs to be summarized.

    model : str, optional (default="gpt-4")
        The name of the OpenAI model to be used for summarization. Defaults to "gpt-4".

    Returns
    -------
    str
        A concise summary of the input documents.

    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f'Create summary of the documents.\n\nDocuments:\n"""\n{documents}\n"""\n\nTopic:',
            },
        ],
        temperature=0,
        top_p=1,
    )
    return response["choices"][0]["message"]["content"]


def plot_circle_chart(values: list[float], labels: list[str], title: str) -> None:
    """
    Plot a circle (pie) chart using the given values and labels.

    Parameters
    ----------
    values : list[float]
        A list of numeric values to represent in the circle chart.

    labels : list[str]
        A list of string labels corresponding to each value in the 'values' list.

    title : str
        The title for the circle chart.

    Returns
    -------
    None

    Notes
    -----
    This function uses matplotlib's pie chart plotting capabilities to render the chart.
    The circle chart starts at a 90-degree angle, and a legend is displayed on the right side of the chart.
    The percentage of each section is displayed on the chart.

    """
    plt.figure(figsize=(7, 10))
    plt.pie(values, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.legend(labels, loc="right", bbox_to_anchor=(1.7, 0.5))
    plt.show()


def preprocess_product_data(df: pd.DataFrame, vm: pd.DataFrame, top_products: int = 6) -> tuple[list[float], list[str]]:
    """
    Preprocess product data to consolidate and get top products based on their occurrences.

    This function aggregates product data from multiple columns, calculates the percentage of each
    product's occurrence, and returns a list of percentage values and labels for the top products.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing product data in columns "product_1", "product_2", and "product_3".

    vm : pd.DataFrame
        A value mapping dataframe that maps product codes to their respective names.

    top_products : int, optional (default=6)
        The number of top products to consider for output.

    Returns
    -------
    tuple[list[float], list[str]]
        A tuple containing two lists:
        1. List of percentage values of the top products.
        2. List of labels corresponding to the top products. The last label is always "OTHERS", representing the
        combined percentage of all other products.

    Notes
    -----
    It's assumed that the product columns ("product_1", "product_2", and "product_3") have numeric codes and that
    the 'vm' dataframe provides a mapping from these codes to product names.

    """
    p1 = df["product_1"].tolist()
    p2 = df["product_2"].tolist()
    p3 = df["product_3"].tolist()

    for p in p2:
        if p != 0:
            p1.append(p)

    for p in p3:
        if p != 0:
            p1.append(p)

    count_dict = Counter(p1)
    total_count = sum(count_dict.values())
    sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    for item, count in sorted_dict.items():
        percentage = (count / total_count) * 100
        sorted_dict[item] = percentage

    values = []
    labels = []
    s = 0.0
    for i, (item, count) in enumerate(sorted_dict.items()):
        if i < top_products:
            values.append(count)
            s = s + count
            labels.append(vm["product_1"][str(item)].split("-")[1])
    values.append(100 - s)
    labels.append(" OTHERS")

    return values, labels


def preprocess_location_data(df: pd.DataFrame, top_products: int = 6) -> tuple[list[float], list[str]]:
    """
    Preprocess location data to consolidate and get top locations based on their occurrences.

    This function aggregates location data from the "location" column, calculates the percentage of each
    location's occurrence, and returns a list of percentage values and labels for the top locations.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe containing location data in the "location" column.

    top_products : int, optional (default=6)
        The number of top locations to consider for output.

    Returns
    -------
    tuple[list[float], list[str]]
        A tuple containing two lists:
        1. List of percentage values of the top locations.
        2. List of labels corresponding to the top locations.

    Notes
    -----
    It's assumed that the "location" column contains categorical or string data representing different locations.

    """
    p1 = df["location"].tolist()

    count_dict = Counter(p1)
    total_count = sum(count_dict.values())
    sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    for item, count in sorted_dict.items():
        percentage = (count / total_count) * 100
        sorted_dict[item] = percentage

    values = []
    labels = []
    for i, (item, count) in enumerate(sorted_dict.items()):
        if i < top_products:
            values.append(count)
            labels.append(item)

    return values, labels


def get_embedding_local(text: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> np.ndarray:
    """
    Extract the embedding for a given text using a pre-trained model and tokenizer.

    Parameters
    ----------
    text : str
        The input text to be embedded.
    tokenizer : PreTrainedTokenizer
        The pre-trained tokenizer used for tokenization of the input text.
    model : PreTrainedModel
        The pre-trained model used to generate embeddings.

    Returns
    -------
    np.ndarray
        The embedding vector for the input text. Specifically, it returns the [CLS] token's embedding
        as the representation of the entire sentence.

    Notes
    -----
    The function processes the text using the given tokenizer and then uses the pre-trained model
    to obtain the embeddings. It specifically extracts the embedding of the [CLS] token as a representation
    of the entire input text. The function ensures no gradients are computed during the embedding extraction
    by using the `torch.no_grad()` context manager.

    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Using the [CLS] embedding as the sentence representation
    return outputs["last_hidden_state"][:, 0, :].squeeze().numpy()


def similarity_score(text1: str, text2: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> float:
    """
    Compute the cosine similarity score between two texts using their embeddings from a pre-trained model.

    Parameters
    ----------
    text1 : str
        The first input text.
    text2 : str
        The second input text.
    tokenizer : PreTrainedTokenizer
        The pre-trained tokenizer used for tokenization.
    model : PreTrainedModel
        The pre-trained model used to generate embeddings.

    Returns
    -------
    float
        The cosine similarity score between the two texts. A value closer to 1 indicates higher similarity
        and a value closer to 0 indicates lower similarity.

    Notes
    -----
    The function obtains embeddings for each text using the `get_embedding_local` function. It then computes
    the cosine distance between the two embeddings and returns its complement (1 - cosine distance) as the
    cosine similarity.

    """
    emb1 = get_embedding_local(text1, tokenizer, model)
    emb2 = get_embedding_local(text2, tokenizer, model)
    # 1 - cosine distance will give cosine similarity
    return 1 - cosine(emb1, emb2)


def find_best_model(
    mod_tok: list[tuple[PreTrainedModel, PreTrainedTokenizer]],
    questions: dict,
    df_all: pd.DataFrame,
    max_rows: int,
    tokenizer_emb: PreTrainedTokenizer,
    model_emb: PreTrainedModel,
) -> int:
    """
    Search for the best model.

    Search for the best model amongst a list of models based on their performance in answering questions
    pertaining to specific contexts from a DataFrame of narratives.

    Parameters
    ----------
    mod_tok : list[tuple[PreTrainedModel, PreTrainedTokenizer]]
        A list of tuples where each tuple contains a pre-trained model and its corresponding tokenizer.
    questions : dict
        A dictionary of questions with context as the key and the question as the value.
    df_all : pd.DataFrame
        A DataFrame containing narratives from which questions are posed to the models.
    max_rows : int
        The maximum number of rows to consider from df_all.
    tokenizer_emb : PreTrainedTokenizer
        The pre-trained tokenizer used for generating embeddings.
    model_emb : PreTrainedModel
        The pre-trained model used for generating embeddings.

    Returns
    -------
    int
        The index of the best model in the mod_tok list based on its performance in answering questions.

    Notes
    -----
    The function iteratively poses questions from the questions dictionary to each model in the mod_tok list.
    The answers generated by the model are then compared to reference answers from another DataFrame using a
    cosine similarity measure. The best model is identified based on the average similarity score of its answers
    to the reference answers.

    """
    idx_model = -1
    best_idx = -1
    best_mean = 0.0
    for model, tokenizer in mod_tok:
        idx_model = idx_model + 1
        time.sleep(1)
        questions_df = {
            "body_part": [],
            "product_1": [],
        }

        # iterate over all cases
        for i, (case_number, narrative) in tqdm(
            enumerate(zip(df_all["cpsc_case_number"].iloc[:max_rows], df_all["text"].iloc[:max_rows])),
            total=len(df_all.iloc[:max_rows]),
            position=0,
            leave=True,
        ):
            # ask questions
            for context, question in questions.items():
                # create questions and tokenize them
                input_text = narrative + "\n" + question
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

                # get answer and remove special tokens
                outputs = model.generate(input_ids, max_new_tokens=20)
                answer = tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "")

                questions_df[context].append(answer)

        questions_df = pd.DataFrame(questions_df)

        v1 = questions_df["body_part"].tolist()
        v2 = questions_df["product_1"].tolist()
        t1 = df_final_1["body_part"].apply(lambda x: x.split("-")[1][1:].lower()).tolist()
        t2 = df_final_1["product_1"].apply(lambda x: x.split("-")[1][1:].lower()).tolist()

        res_1 = []
        for e1, e2 in tqdm(zip(v1, t1[:max_rows]), total=len(t1[:max_rows])):
            score = similarity_score(e1, e2, tokenizer_emb, model_emb)
            res_1.append(score)

        res_2 = []
        for e1, e2 in tqdm(zip(v2, t2[:max_rows]), total=len(t2[:max_rows])):
            score = similarity_score(e1, e2, tokenizer_emb, model_emb)
            res_2.append(score)

        mean_1 = np.mean(res_1)
        mean_2 = np.mean(res_2)
        print(mean_1)
        print(mean_2)

        if mean_1 + mean_2 > best_mean:
            best_idx = idx_model
            best_mean = mean_1 + mean_2

    return best_idx


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    questions: dict[str, str],
    df_all: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """
    Generate answers to a set of questions based on narratives from a DataFrame using a pre-trained model.

    Function inspired by: https://www.drivendata.org/competitions/217/cdc-fall-narratives/community-code/13/

    Parameters
    ----------
    model : PreTrainedModel
        The pre-trained model used for generating answers.
    tokenizer : PreTrainedTokenizer
        The pre-trained tokenizer used for tokenizing the input text.
    questions : dict[str, str]
        A dictionary of questions with context as the key and the question as the value.
    df_all : pd.DataFrame
        A DataFrame containing narratives from which questions are posed to the model.
    max_rows : int
        The maximum number of rows to consider from df_all.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the answers generated by the model for each context in the questions dictionary.

    Notes
    -----
    The function iteratively poses questions from the questions dictionary to the model for each narrative
    in the df_all DataFrame. The answers generated by the model are then collected and returned as a DataFrame.

    """
    answers_df = {k: [] for k in questions}
    for i, (case_number, narrative) in tqdm(
        enumerate(zip(df_all["cpsc_case_number"].iloc[:max_rows], df_all["text"].iloc[:max_rows])),
        total=len(df_all.iloc[:max_rows]),
        position=0,
        leave=True,
    ):
        # ask questions
        for context, question in questions.items():
            # create questions and tokenize them
            input_text = narrative + "\n" + question
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

            # get answer and remove special tokens
            outputs = model.generate(input_ids, max_new_tokens=20)
            answer = tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "")
            answers_df[context].append(answer)

    answers_df = pd.DataFrame(answers_df)

    return answers_df


def lem_sample(text: str, nlp) -> list[str]:
    """
    Lemmatize a given text and filters it based on specific part-of-speech tags.

    Parameters
    ----------
    text : str
        The input text to be lemmatized.
    nlp :
        The loaded spacy model to perform lemmatization.

    Returns
    -------
    list[str]
        A list containing lemmatized words from the input text that match the allowed part-of-speech tags.

    Notes
    -----
    The function uses a predefined set of allowed part-of-speech tags (NOUN, ADJ, VERB, ADV) to filter
    the lemmatized words. Only words with these tags are returned in the output list.

    """
    # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # only keep certain kinds of words
    allowed_tags = ["NOUN", "ADJ", "VERB", "ADV"]
    text = nlp(text[1:])
    text = [token.lemma_ for token in text if token.pos_ in allowed_tags]
    text = text[0] if text else ""
    return text


def preprocess_answers(
    input_data: str, top_products: int, if_why: bool, nlp
) -> tuple[list[float], list[str], list[str]]:
    """
    Preprocess and aggregates answers from the input data.

    Parameters
    ----------
    input_data : str
        The raw input data containing answers to be processed.
    top_products : int
        The number of top products (or answers) to consider for aggregation.
    if_why : bool
        A flag to decide if the preprocessing involves removing the first character or lemmatization.
    nlp :
        The loaded spacy model to perform lemmatization, if required.

    Returns
    -------
    tuple[list[float], list[str], list[str]]
        - A list of aggregated percentages of top answers.
        - A list of labels corresponding to the top answers.
        - A list of processed input data based on the `if_why` flag.

    Notes
    -----
    The function first decides the preprocessing step based on the `if_why` flag. If `if_why` is True,
    it simply removes the first character from each entry in `input_data`. Otherwise, it lemmatizes
    the entries. The function then counts the occurrences of each unique entry, calculates its
    percentage, and returns the percentages and labels of the top answers as specified by the
    `top_products` parameter.

    """
    if if_why:
        input_data_2 = [t[1:] for t in input_data]
    else:
        input_data_2 = [lem_sample(t, nlp) for t in tqdm(input_data)]
        # input_data_2 = lem_sample(input_data)
    print(len(np.unique(input_data_2)))

    count_dict = Counter(input_data_2)
    total_count = sum(count_dict.values())
    sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    for item, count in sorted_dict.items():
        percentage = (count / total_count) * 100
        sorted_dict[item] = percentage

    values = []
    labels = []
    for i, (item, count) in enumerate(sorted_dict.items()):
        if i < top_products:
            values.append(count)
            labels.append(item)

    return values, labels, input_data_2


def prepare_stats_to_plot(
    tp_2: pd.DataFrame, t1: str, t2: str, top_products: int = 9
) -> dict[str, tuple[list[float], list[str]]]:
    """
    Prepare statistical data for plotting by aggregating the top categories based on a given threshold.

    Parameters
    ----------
    tp_2 : pd.DataFrame
        The DataFrame containing the data to be processed.
    t1 : str
        The primary column in the DataFrame based on which unique categories are extracted.
    t2 : str
        The secondary column containing the data to be aggregated for each unique category from `t1`.
    top_products : int, optional (default is 9)
        The number of top categories to consider for aggregation in the resulting data.

    Returns
    -------
    dict[str, tuple[list[float], list[str]]]
        A dictionary where each key represents a unique category from the `t1` column,
        and the associated value is a tuple containing:
        - A list of aggregated percentages of top categories from the `t2` column.
        - A list of labels corresponding to these top categories.

    Notes
    -----
    For each unique category in the `t1` column, the function aggregates the occurrences of each
    category in the `t2` column, calculates its percentage, and then retains only the top categories
    as specified by the `top_products` parameter. All other categories are grouped into a single
    "zothers" category.

    """
    stats_data = tp_2[t1].unique().tolist()
    data = {ld: [] for ld in stats_data}
    for k, v in data.items():
        x_in = tp_2[["action", "why"]][tp_2[t1] == k]
        # print(x_in.shape)
        input_data_2 = x_in[t2].tolist()

        count_dict = Counter(input_data_2)
        total_count = sum(count_dict.values())
        sorted_dict = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
        for item, count in sorted_dict.items():
            percentage = (count / total_count) * 100
            sorted_dict[item] = percentage

        values = []
        labels = []
        for i, (item, count) in enumerate(sorted_dict.items()):
            if i < top_products:
                values.append(count)
                labels.append(item)

        labels.append("zothers")
        values.append(100 - np.sum(values))

        data[k] = (values, labels)

    return data


def plot_stats(
    data: dict[str, tuple[list[float], list[str]]],
    x_label: str,
    y_label: str,
    title: str,
    x_size: int,
    y_size: int,
    bar_width: float = 0.5,
    horizontal: bool = False,
) -> None:
    """
    Plot a stacked bar chart based on given statistical data.

    Parameters
    ----------
    data : dict[str, tuple[list[float], list[str]]]
        A dictionary containing data to be plotted. The key represents the x-axis label (like month) and the
        value is a tuple containing:
        - A list of percentages.
        - A list of labels corresponding to these percentages.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    title : str
        Title for the plot.
    x_size : int
        Width of the plot figure.
    y_size : int
        Height of the plot figure.
    bar_width : float, optional (default is 0.5)
        Width of the individual bars.
    horizontal : bool, optional (default is False)
        If set to True, the bars are plotted horizontally; otherwise, they are plotted vertically.

    Returns
    -------
    None

    Description
    -----------
    The function constructs a stacked bar chart where each bar represents a key from the input dictionary.
    Each segment of the bar is colored differently based on the labels in the tuple associated with the key.
    The segments are stacked in the order of appearance of the labels in the `data` dictionary.
    The x-axis represents the keys from the dictionary, while the y-axis represents the accumulated percentage.
    The function can generate both vertical and horizontal stacked bars based on the `horizontal` parameter.

    """
    x_labels_2 = list(data.keys())
    x_labels = [xl.replace("/", "\n") for xl in x_labels_2]

    unique_labels = set()
    for _, labels in data.values():
        unique_labels.update(labels)
    unique_labels = sorted(list(unique_labels))

    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    r = np.arange(len(data))
    plt.figure(figsize=(x_size, y_size))

    bottoms = np.zeros(len(data))

    for label in unique_labels:
        percentages = []
        for month_data in data.values():
            if label in month_data[1]:
                idx = month_data[1].index(label)
                percentages.append(month_data[0][idx])
            else:
                percentages.append(0)

        if horizontal:
            plt.barh(
                r,
                percentages,
                left=bottoms,
                color=label_to_color[label],
                height=bar_width,
                edgecolor="grey",
                label=label,
            )
        else:
            plt.bar(
                r,
                percentages,
                bottom=bottoms,
                color=label_to_color[label],
                width=bar_width,
                edgecolor="grey",
                label=label,
            )
        bottoms += percentages

    if horizontal:
        plt.ylabel(x_label, fontweight="bold")
        plt.xlabel(y_label, fontweight="bold")
        plt.yticks(list(range(len(x_labels))), x_labels)
        plt.xlim(0, 100)
    else:
        plt.xlabel(x_label, fontweight="bold")
        plt.ylabel(y_label, fontweight="bold")
        plt.xticks(list(range(len(x_labels))), x_labels)
        plt.ylim(0, 100)

    plt.title(title, fontweight="bold")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_histogram(values: list[float | str], title: str, x_label: str, y_label: str, bins: int = 10) -> None:
    """
    Generate and displays a histogram for the given set of values.

    The function takes in a list of values, plot settings such as title, x-label, y-label, and bin size.
    It then creates and displays the histogram using the provided settings.

    Parameters
    ----------
    values : list[float | str]
        List of numerical or string values for which the histogram is to be plotted.

    title : str
        Title of the histogram.

    x_label : str
        Label for the x-axis.

    y_label : str
        Label for the y-axis.

    bins : int, optional (default=10)
        Number of bins to be used in the histogram.

    Returns
    -------
    None
        The function displays the histogram and does not return any value.

    """
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def prepare_time_data(x: list[float]) -> tuple[list[str], list[str], list[str]]:
    """
    Convert a list of date strings into lists of days, weeks, and months.

    Parameters
    ----------
    x : list[float]
        A list of date strings in the format "YYYY-MM-DD".

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Three lists containing:
        1. The day of the week corresponding to each date.
        2. The week number (starting on Monday) of the year corresponding to each date.
        3. The month number corresponding to each date.

    Notes
    -----
    The function uses Python's datetime module to process and format the date strings.
    A progress bar is displayed using the tqdm module as each date string is processed.

    """
    days = []
    weeks = []
    months = []
    for e in tqdm(x):
        date_string = e
        date_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        day_of_week = date_object.strftime("%A")
        week_number_monday = date_object.strftime("%W")
        month_number = date_object.strftime("%m")

        days.append(day_of_week)
        weeks.append(week_number_monday)
        months.append(month_number)

    return days, weeks, months


def plot_histogram_perc(values: list[float | str], labels: list[str], title: str, x_label: str, y_label: str) -> None:
    """
    Plot a histogram showing the percentage of each label based on the given values.

    Parameters
    ----------
    values : list[float | str]
        List of values which might be either float or strings. These values are the instances for which the
        frequency is to be calculated.
    labels : list[str]
        List of unique labels based on which the frequency will be calculated from the values.
    title : str
        Title of the histogram.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.

    Returns
    -------
    None

    Description
    -----------
    The function calculates the frequency (in percentage) of each label present in the 'labels' list based on
    how many times they appear in the 'values' list. The histogram then plots these frequencies against each label.
    The y-axis is formatted to display percentages.

    """
    # Calculate the frequency of each label
    freqs = [100 * values.count(label) / len(values) for label in labels]

    # Plot the histogram based on label order
    plt.bar(labels, freqs, align="center")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)  # Optional: To make labels more readable in case of longer strings
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.show()


if __name__ == "__main__":
    # set up
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("words")
    sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

    tqdm.pandas()
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    cuda_v = torch.version.cuda if device != "cpu" else ""
    f"Running on device: '{device}' with CUDA version '{cuda_v}."

    # load data
    df_1 = pd.read_csv("data/primary_data.csv")
    vm_1 = json.load(open("data/variable_mapping.json"))

    # preprocess narrative
    df_1["text"] = df_1["narrative"].apply(clean_narrative)
    print(df_1.iloc[1].T)

    # fill variables
    df_final_1 = fill_variables(df_1, vm_1)
    print(df_final_1.iloc[1].T)

    # create untext
    create_untext(df_final_1)

    # create untext embedding
    openai.api_key = "your_key"
    x_1 = df_final_1["untext"].tolist()
    embed_1 = create_embedding(x_1, "untext_embed_final_oo")
    print(embed_1.shape)

    # search optimal parameters
    best_param_use_1, best_clusters_use_1, trials_use_1 = search_param(embed_1)

    # create clusters
    clusters_1, umap_embeddings_1 = generate_clusters(
        embed_1, n_neighbors=13, n_components=3, min_cluster_size=100, random_state=31
    )
    label_count_1, cost_1 = score_clusters(clusters_1)
    print(label_count_1, cost_1)
    print(np.sum(clusters_1.labels_ == -1))

    # plot clusters
    fig_1a = create_plot_3d(umap_embeddings_1, clusters_1, label_count_1, True, "Clusters for 'untext' data")
    # fig_1a.show()

    # plot cluster without noise
    fig_1b = create_plot_3d(umap_embeddings_1, clusters_1, label_count_1, False,
                            "Clusters for 'untext' data without noise")

    # generate keywords
    df_final_1["untext"] = df_final_1["untext"].apply(lambda x: str(x))
    for idx in [0]:
        print()
        lbs = clusters_1.labels_.tolist()
        vrs = [i for i in range(len(lbs)) if lbs[i] == idx]
        print(len(vrs))

    for i in range(5):
        print(df_final_1["untext"].iloc[vrs[i]])

    all_text_1 = " ".join(df_final_1["untext"])

    keywords = gen_keywords(all_text_1)
    for kw in keywords:
        print(kw)

    # create embedding for whole narrative
    x_2 = df_final_1["text"].tolist()
    embed_2 = create_embedding(x_2, filename="text_embed_final_oo")
    print(embed_2.shape)

    # search optimal parameters
    best_param_use_2, best_clusters_use_2, trials_use_2 = search_param(embed_2)

    # create clusters
    clusters_2, umap_embeddings_2 = generate_clusters(
        embed_2, n_neighbors=5, n_components=3, min_cluster_size=100, random_state=31
    )
    label_count_2, cost_2 = score_clusters(clusters_2)
    print(label_count_2, cost_2)
    print(np.sum(clusters_2.labels_ == -1))

    # plot clusters
    fig_2a = create_plot_3d(umap_embeddings_2, clusters_2, label_count_2, True, "Clusters for 'text' data")
    # fig_2a.show()

    # plot clusters without noise
    fig_2b = create_plot_3d(umap_embeddings_2, clusters_2, label_count_2, False,
                            "Clusters for 'text' data without noise")
    # fig_2b.show()

    # Save clusters
    with open("clus_data.pkl", "wb") as f:
        pickle.dump([clusters_1, clusters_2, umap_embeddings_1, umap_embeddings_2], f)

    # create summary of clusters
    summary_clusters = {},
    for idx in range(80):
        lbs = clusters_2.labels_.tolist()
        vrs = [i for i in range(len(lbs)) if lbs[i] == idx]
        docs = "\n".join(df_final_1["text"].iloc[vrs[:30]].values)
        desc = create_summary(docs)
        summary_clusters[idx] = [desc, len(vrs)]
        time.sleep(1)

    # Save summary of clusters
    with open("summary_clusters.pkl", "wb") as f:
        pickle.dump(summary_clusters, f)

    # sample summary
    for i in [0, 1, 5, 6]:
        print(summary_clusters[i])

    # generate keywords
    all_text_2 = " ".join(df_final_1["text"])
    keywords = gen_keywords(all_text_2)
    for kw in keywords:
        print(kw)

    # plot most common items
    values_1, labels_1 = preprocess_product_data(df_1, vm_1)
    plot_circle_chart(values_1, labels_1, "Percentage of things connected with falls")

    # plot most common locations
    values_2, labels_2 = preprocess_location_data(df_1)
    plot_circle_chart(values_2, labels_2, "Percentage of locations where falls occur")

    # extract perticipating event
    test_questions = {
        "body_part": "What body part with the most severe injury was affected in the fall?",
        "product_1": "What consumer product was involved in the incident?",
    }

    mod_tok_list = [
        (
            T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map=device),
            T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False),
        ),
        (
            MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp", device_map=device),
            MvpTokenizer.from_pretrained("RUCAIBox/mvp"),
        ),
        (
            BertLMHeadModel.from_pretrained("bert-base-uncased", device_map=device),
            AutoTokenizer.from_pretrained("bert-base-uncased"),
        ),
    ]

    # Initialize tokenizer and model
    tokenizer_b = BertTokenizer.from_pretrained("bert-base-uncased")
    model_b = BertModel.from_pretrained("bert-base-uncased")

    # find best model
    best_model_idx = find_best_model(mod_tok_list, test_questions, df_final_1, 3, tokenizer_b, model_b)

    best_model, best_tokenizer = mod_tok_list[best_model_idx]

    # generate answers
    final_questions = {
        # "action": "What patient's activity took place during the fall?",
        "action": "What was the patient doing at the time of the incident?",
        "type": "What kind of fall took place?",
        "who": "Who has fallen?",
        "what": "What happened to the patient during the incident?",
        "why": "What caused the patient to fall?",
        "when": "When did the fall occur?",
        "where": "Where did the fall occur?",
        "how": "How did the fall occur?",
    }
    ans_df_1 = generate_answers(best_model, best_tokenizer, final_questions, df_final_1, 10)
    print(ans_df_1.head())

    # extract item and action
    type_questions = {
        "action": "What was the patient doing before the fall? Return only one verb.",
        "why": "What item or place, had contact with the patient's body during incident? Return only one noun.",
    }
    ans_df_2 = generate_answers(best_model, best_tokenizer, type_questions, df_final_1, 27)
    print(ans_df_2.head())

    # actions percentage stats
    ans_df_3 = pd.read_csv("ans_df_3.csv")
    print(ans_df_3.shape)
    ans_actions = ans_df_3["action"].tolist()
    ans_actions_2 = [str(w) for w in ans_actions]
    # action_text = ",".join(ans_actions_2)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    values_a1, labels_a1, actions = preprocess_answers(ans_actions_2, 10, False, nlp)
    plot_circle_chart(values_a1, labels_a1, "Actions")

    # causes percentage stats
    ans_why = ans_df_3["why"].tolist()
    values_a2, labels_a2, whys = preprocess_answers(ans_why, 10, True, nlp)
    plot_circle_chart(values_a2, labels_a2, "Why")

    # fall type by different groups
    tp_1 = df_final_1[["treatment_date", "age", "sex", "race", "diagnosis", "body_part", "disposition", "location",
                       "alcohol", "drug"]]
    tp_2 = tp_1.copy()
    tp_2["action"] = actions
    tp_2["why"] = whys
    print(tp_2.shape)
    tp_2["action"] = tp_2["action"].apply(lambda x: str(x))
    tp_2["why"] = tp_2["why"].apply(lambda x: str(x))
    print(tp_2.head())
    print(tp_2[["sex", "race", "diagnosis", "body_part", "disposition", "location", "alcohol", "drug"]].describe())

    # action type
    t1, t2 = "sex", "action"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 6, 5, bar_width=0.5)

    # cause type
    t1, t2 = "sex", "why"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 6, 5, bar_width=0.5)

    # action type by race
    t1, t2 = "race", "action"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 12, 5, bar_width=0.5)

    # cause type by race
    t1, t2 = "race", "why"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 12, 5, bar_width=0.5)

    # action type by disposition
    t1, t2 = "disposition", "action"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 5, bar_width=0.5)

    # cause type by disposition
    t1, t2 = "disposition", "why"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 5, bar_width=0.5)

    # action type by body part
    t1, t2 = "body_part", "action"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 15, bar_width=0.5, horizontal=True)

    # cause type by body part
    t1, t2 = "body_part", "why"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 15, bar_width=0.5, horizontal=True)

    # action by diagnosis
    t1, t2 = "diagnosis", "action"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 15, bar_width=0.5, horizontal=True)

    # cause by diagnosis
    t1, t2 = "diagnosis", "why"
    data = prepare_stats_to_plot(tp_2, t1, t2)
    plot_stats(data, f"Groups of {t1}", "Percentage usage", t1, 15, 15, bar_width=0.5, horizontal=True)

    # check time patterns
    df_sup = pd.read_csv("data/supplementary_data.csv")
    vm_1 = json.load(open("data/variable_mapping.json"))
    df_sup_1 = fill_variables(df_sup, vm_1)
    print(df_sup_1.shape)

    # age distribution
    ages = df_sup_1["age"].tolist()
    plot_histogram(ages, "Age distribution", "Age", "Number of samples", bins=100)

    treat_dates = df_sup_1["treatment_date"].tolist()
    days_1, weeks_1, months_1 = prepare_time_data(treat_dates)

    # time distributions
    plot_histogram(days_1, "Days of week distribution", "Day of week", "Number of samples", bins=7)
    plot_histogram(weeks_1, "Weeks of year distribution", "Week of year", "Number of samples", bins=52)
    plot_histogram(months_1, "Months distribution", "Month", "Number of samples", bins=12)

    # time trends by specific group
    l_d = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    l_m = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
