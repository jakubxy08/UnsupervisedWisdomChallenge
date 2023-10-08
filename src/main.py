"""Analysis of unsupervised wisdom data."""

import json
import re

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
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
