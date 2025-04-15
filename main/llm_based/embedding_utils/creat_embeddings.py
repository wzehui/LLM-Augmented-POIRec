import pandas as pd
import json
import os.path as osp
from main.llm_based.embedding_utils.openai_utils import set_embeddings_from_df

WORKING_DIR = "../../../yelp"
# product_lookup = pd.read_csv(f"{WORKING_DIR}/csv/business.csv", usecols=[
#     'name', 'categories'])
product_lookup = pd.read_csv(f"{WORKING_DIR}/csv/business_filtered.csv",
                             usecols=['business_id', 'name', 'categories'])
product_lookup_empty_names = product_lookup[~product_lookup["name"].notnull()]
empty_name_count = product_lookup[product_lookup["name"].isnull()].shape[0]
product_lookup = product_lookup[product_lookup["name"].notnull()]


json_path = osp.join(WORKING_DIR, "dataset/itemid_to_int.json")
with open(json_path, "r") as f:
    itemid_to_int = json.load(f)
product_lookup["ItemId"] = product_lookup["business_id"].map(itemid_to_int)
product_lookup.drop(columns=["business_id"], inplace=True)

# product_lookup['formatted_text'] = product_lookup['name'] + ' - ' + product_lookup['categories'].fillna('')
product_lookup['formatted_text'] = product_lookup['categories'].fillna('')

product_lookup.drop(columns=["name", "categories"], inplace=True)
product_lookup.rename(columns={"formatted_text": "combined_text"}, inplace=True)

import tiktoken
# Load OpenAI's tokenizer for "text-embedding-ada-002"
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
# Compute token count for each input
product_lookup['token_count'] = product_lookup['combined_text'].apply(lambda x: len(tokenizer.encode(x)))
# Calculate average token count
average_tokens = product_lookup['token_count'].mean()
print(f"Average token count per input: {average_tokens:.2f}")

step_size = 1000
batch_product_lookup = product_lookup.iloc[0:step_size]
cur_step = step_size
processed_batches = []
num_batches_processed = 0
while not(batch_product_lookup.empty):
    print(f"Currently at {cur_step} of {len(product_lookup)}")
    batch_product_lookup = set_embeddings_from_df(batch_product_lookup)
    processed_batches.append(batch_product_lookup)

    batch_product_lookup = product_lookup.iloc[cur_step:cur_step + step_size]
    cur_step += step_size

new_product_lookup = pd.concat(processed_batches, copy=False)
new_product_lookup["ada_embedding"] = new_product_lookup["ada_embedding"].apply(lambda x : json.loads(x) if isinstance(x, str) else x)

emb_dim = len(new_product_lookup.iloc[0]["ada_embedding"])
product_lookup_empty_names["ada_embedding"] = [[0.0 for _ in range(emb_dim)]] * len(product_lookup_empty_names)
product_lookup_empty_names["name"] = "Unknown item"

new_product_lookup = pd.concat([new_product_lookup, product_lookup_empty_names], axis=0)
new_product_lookup.memory_usage(deep=True)

# modify format
new_product_lookup = new_product_lookup.drop(columns=['name', 'business_id', 'categories'])
new_product_lookup["ItemId"] = new_product_lookup["ItemId"].astype(int)
new_product_lookup = new_product_lookup.rename(columns={"ada_embedding": "embedding"})

# new_product_lookup.to_csv(f"{WORKING_DIR}/embeddings/product_embeddings_openai"
#                           f".csv.gz", index=False, compression="gzip")
new_product_lookup.to_csv(f"{WORKING_DIR}/embeddings/product_no_name_embeddings_openai.csv.gz", index=False, compression="gzip")

new_product_lookup["length"] = new_product_lookup["embedding"].apply(len)
new_product_lookup["type"] = new_product_lookup["embedding"].apply(lambda x : type(x))

print(new_product_lookup["length"].unique())
print(new_product_lookup["type"].unique())


