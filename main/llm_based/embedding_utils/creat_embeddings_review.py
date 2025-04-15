import pandas as pd
import json
import os.path as osp
from main.llm_based.embedding_utils.openai_utils import set_embeddings_from_df

WORKING_DIR = "../../../yelp"
# OUTPUT = f"{WORKING_DIR}/embeddings/review_summary_embeddings_deepseek.csv.gz"
OUTPUT = (f"{WORKING_DIR}/embeddings/review_summary_no_rating_embeddings_deepseek.csv.gz")

product_lookup = pd.read_csv(f"{WORKING_DIR}/csv/business_filtered.csv",
                             usecols=['business_id', 'stars', 'name',
                                      'categories'])
review_summary =  pd.read_csv(f""
                              f"{WORKING_DIR}/csv/review_summary_deepseek.csv",
                              encoding="utf-8-sig")

review_summary = review_summary.merge(product_lookup, on="business_id", how="left")

review_summary["combined_text"] = review_summary.apply(
    lambda row:
    # f"Sentiment Score: {row['sentiment_score']:.2f} (Confidence: {row['sentiment_confidence']:.0%}) | "
                # f"Rating: {row['stars']} / 5 | "
                f"Summary: {row['summary']} | Keywords: {row['keywords']} | Themes: {row['themes']}",
    axis=1
)

import tiktoken
# Load OpenAI's tokenizer for "text-embedding-ada-002"
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
# Compute token count for each input
review_summary['token_count'] = review_summary['combined_text'].apply(lambda x: len(tokenizer.encode(x)))
# Calculate average token count
average_tokens = review_summary['token_count'].mean()
print(f"Average token count per input: {average_tokens:.2f}")

step_size = 1000
batch_review_summary = review_summary.iloc[0:step_size]
cur_step = step_size
processed_batches = []
num_batches_processed = 0

while not batch_review_summary.empty:
    print(f"Currently processing {cur_step} of {len(review_summary)}")
    batch_review_summary = set_embeddings_from_df(batch_review_summary, text_column="combined_text")
    processed_batches.append(batch_review_summary)
    batch_review_summary = review_summary.iloc[cur_step:cur_step + step_size]
    cur_step += step_size

new_review_summary = pd.concat(processed_batches, copy=False)

new_review_summary["ada_embedding"] = new_review_summary["ada_embedding"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else x
)

json_path = osp.join(WORKING_DIR, "dataset/itemid_to_int.json")
with open(json_path, "r") as f:
    itemid_to_int = json.load(f)
new_review_summary["ItemId"] = new_review_summary["business_id"].map(itemid_to_int)
new_review_summary.drop(columns=["business_id"], inplace=True)

# Drop unnecessary columns
new_review_summary = new_review_summary.drop(['sentiment_score', 'sentiment_confidence', 'summary', 'keywords',
              'themes', 'name', 'stars', 'categories', 'combined_text'], axis=1)
# Rename embedding column
new_review_summary = new_review_summary.rename(columns={"ada_embedding": "embedding"})
# Ensure ItemId is an integer if it exists
new_review_summary["ItemId"] = new_review_summary["ItemId"].astype(int)

# Validate embedding dimensions and types
lengths = new_review_summary["embedding"].apply(len).unique()
types = new_review_summary["embedding"].apply(lambda x: type(x)).unique()
print(lengths)  # Ensure all embeddings have the same dimensions
print(types)  # Ensure all embeddings are lists or correct data types

if len(lengths) == 1 and all(t == list for t in types):
    new_review_summary.to_csv(OUTPUT, index=False, compression="gzip")
    print(f"Embeddings saved to {OUTPUT}")
else:
    print("Error: Inconsistent embedding dimensions or types. File not saved.")