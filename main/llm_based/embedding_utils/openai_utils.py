import pickle
import pandas as pd
import os
import openai

EMBEDDING_ENGINE = "text-embedding-ada-002"
COMPLETION_ENGINE = "text-ada-001"

# Get path to current folder.
cur_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

# OpenAI API authentication
KEY_LINE = 1  # 1-5
with open("./key.txt", "r") as file:
    lines = file.readlines()
    api_key = lines[KEY_LINE - 1].strip()
client = openai.OpenAI(api_key=api_key)

# Establish a cache of embeddings to avoid recomputing
embedding_cache_path = f"{cur_folder}openai_embeddings_cache.pkl"

# Load the cache if it exists, otherwise create an empty one
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    if "y" not in input(
            "Have you ensured that you downloaded the embedding cache from GCP? "):
        raise Exception("No embedding cache found.")
    embedding_cache = {}
    with open(embedding_cache_path, "wb") as embedding_cache_file:
        pickle.dump(embedding_cache, embedding_cache_file)


def in_cache(string: str):
    return (string, EMBEDDING_ENGINE) in embedding_cache.keys()


def prompt_openai(prompt: str, **kwargs) -> str:
    return openai.Completion.create(
        model=COMPLETION_ENGINE,
        prompt=prompt,
        **kwargs,
    )["choices"][0]["text"].strip()


def get_openai_embedding(text: str, model="text-embedding-ada-002") -> list:
    """Retrieve OpenAI embedding for a single text using the new API."""
    response = client.embeddings.create(
        input=[text],  # Must be a list
        model=model
    )
    return response.data[0].embedding


def get_openai_embeddings(texts: list, model="text-embedding-ada-002") -> list:
    """Retrieve OpenAI embeddings for a list of texts using the new API."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]


def embedding_from_string(string: str) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, EMBEDDING_ENGINE) not in embedding_cache.keys():
        embedding_cache[(string, EMBEDDING_ENGINE)] = get_openai_embedding(
            string)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, EMBEDDING_ENGINE)]


def set_embeddings_from_df(df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:
    """Set the embeddings for a given text column in the DataFrame."""
    texts = list(df[text_column])

    missing_texts = [text for text in texts if
                     (text, EMBEDDING_ENGINE) not in embedding_cache]

    if missing_texts:
        new_embeddings = get_openai_embeddings(missing_texts)
        for text, embedding in zip(missing_texts, new_embeddings):
            embedding_cache[(text, EMBEDDING_ENGINE)] = embedding

        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)

    embeddings = [embedding_cache[(text, EMBEDDING_ENGINE)] for text in texts]
    df_copy = df.copy()
    df_copy["ada_embedding"] = embeddings

    return df_copy