"""Analysis of unsupervised wisdom data."""
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
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Figure
from tqdm import tqdm, trange

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
