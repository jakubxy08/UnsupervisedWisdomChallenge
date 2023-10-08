"""Analysis of unsupervised wisdom data."""
import json
import os
import pickle
import re
import time
from typing import Any

import hdbscan
import nltk
import numpy as np
import openai
import pandas as pd
import spacy
import torch
import umap
from hdbscan.hdbscan_ import HDBSCAN
from hyperopt import fmin, hp, partial, space_eval, STATUS_OK, tpe, Trials
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
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
