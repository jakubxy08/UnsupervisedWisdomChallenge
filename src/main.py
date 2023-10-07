"""Analysis of unsupervised wisdom data."""

import nltk
import torch
from tqdm import tqdm

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
