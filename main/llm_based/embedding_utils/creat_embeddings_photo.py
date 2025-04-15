import pandas as pd
import json
import os.path as osp
import ast
from main.llm_based.embedding_utils.openai_utils import set_embeddings_from_df

WORKING_DIR = "../../../yelp"
# OUTPUT = f"{WORKING_DIR}/embeddings/photo_summary_embeddings_openai.csv.gz"
OUTPUT = (f"{WORKING_DIR}/embeddings/photo_summary_no_attribute_embeddings_openai.csv.gz")
product_lookup = pd.read_csv(f"{WORKING_DIR}/csv/business_filtered.csv", usecols=['business_id', 'name', 'attributes'])
photo_summary =  pd.read_csv(f"{WORKING_DIR}/csv/photo_summary_openai.csv",
                             encoding="utf-8-sig")
photo_summary = photo_summary.merge(product_lookup, on="business_id", how="right")

def combine_attributes(row):
    parts = []

    if pd.notna(row['summary']):
        parts.append(f"Summary: {row['summary']}")
    if pd.notna(row['keywords']):
        parts.append(f"Keywords: {row['keywords']}")
    if pd.notna(row['indoor_color_tone']):
        parts.append(f"Indoor Color Tone: {row['indoor_color_tone']}")
    if pd.notna(row['venue_style']):
        parts.append(f"Venue Style: {row['venue_style']}")
    if pd.notna(row['food_style']):
        parts.append(f"Food Style: {row['food_style']}")
    if pd.notna(row['drink_style']):
        parts.append(f"Drink Style: {row['drink_style']}")
    if pd.notna(row['target_audience']):
        parts.append(f"Target Audience: {row['target_audience']}")
    if pd.notna(row['special_features']):
        parts.append(f"Special Features: {row['special_features']}")

    return " | ".join(parts)

def safe_eval(value):
    try:
        if isinstance(value, str):
            return ast.literal_eval(value)  # Convert string to dictionary
        elif isinstance(value, dict):
            return value  # Already a dictionary, return as is
        else:
            return {}  # Convert NaN or other types to empty dictionary
    except (ValueError, SyntaxError):
        return {}  # If conversion fails, return empty dictionary

# Function to process attributes and retain only meaningful values
def process_attributes(row):
    flat_dict = {}
    for key, value in row.items():
        parsed_value = safe_eval(value)  # Ensure it's a real dictionary
        if isinstance(parsed_value, dict):  # If it's a nested dictionary
            # Keep only the keys where the value is True
            true_keys = [sub_key for sub_key, sub_value in parsed_value.items() if sub_value is True]
            flat_dict[key] = ", ".join(true_keys) if true_keys else None  # Store as a single column
        else:
            flat_dict[key] = parsed_value  # Keep normal values
    return flat_dict

def combine_normalized_attributes(row):
    parts = []
    for col, value in row.items():
        if pd.isna(value):
            continue
        elif isinstance(value, bool):  # 判断是否是布尔值
            parts.append(f"{col}: {'yes' if value else 'no'}")
        else:
            parts.append(f"{col}: {value}")
    return " | ".join(parts)

photo_summary["photo_text"] = photo_summary.apply(combine_attributes, axis=1)

photo_summary["attributes"] = photo_summary["attributes"].apply(lambda x: safe_eval(x) if pd.notna(x) else {})
photo_summary_normalized = photo_summary["attributes"].apply(process_attributes).apply(pd.Series, dtype=object)

photo_summary["attributes_text"] = photo_summary_normalized.apply(combine_normalized_attributes, axis=1)
# photo_summary["combined_text"] = photo_summary.apply(
#     lambda row: f"{row['photo_text']} | {row['attributes_text']}".strip(" |"),
#     axis=1
# )

photo_summary["combined_text"] = photo_summary.apply(
    lambda row: f"{row['photo_text']} "
                # f"| {row['attributes_text']}".strip(" |")
    ,axis=1
)

import tiktoken
# Load OpenAI's tokenizer for "text-embedding-ada-002"
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
# Compute token count for each input
photo_summary['token_count'] = photo_summary['combined_text'].apply(lambda x: len(tokenizer.encode(x)))
# Calculate average token count
average_tokens = photo_summary['token_count'].mean()
print(f"Average token count per input: {average_tokens:.2f}")

non_empty_photo_summary = photo_summary[photo_summary["combined_text"].notna() & (photo_summary["combined_text"].str.strip() != "")].copy()
empty_photo_summary = photo_summary[photo_summary["combined_text"].isna() | (photo_summary["combined_text"].str.strip() == "")].copy()
print(f"🚀 过滤后 non_empty_photo_summary: {len(non_empty_photo_summary)} 行")
print(f"🚀 过滤后 empty_photo_summary: {len(empty_photo_summary)} 行")

step_size = 1000
batch_photo_summary = non_empty_photo_summary.iloc[0:step_size]
cur_step = step_size
processed_batches = []
num_batches_processed = 0
while not batch_photo_summary.empty:
    print(f"Currently processing {cur_step} of {len(non_empty_photo_summary)}")
    batch_photo_summary = set_embeddings_from_df(batch_photo_summary, text_column="combined_text")
    processed_batches.append(batch_photo_summary)
    batch_photo_summary = non_empty_photo_summary.iloc[cur_step:cur_step + step_size]
    cur_step += step_size

# 合并已处理的 batches
new_photo_summary = pd.concat(processed_batches, copy=False)

# 处理 empty rows：填充全 0 向量
embedding_dim = len(new_photo_summary["ada_embedding"].iloc[0])  # 获取 embedding 维度
empty_photo_summary["ada_embedding"] = empty_photo_summary[
    "combined_text"].apply(lambda x: [0.0] * embedding_dim)

new_photo_summary = pd.concat([new_photo_summary, empty_photo_summary], copy=False)

new_photo_summary["ada_embedding"] = new_photo_summary["ada_embedding"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else x
)

json_path = osp.join(WORKING_DIR, "dataset/itemid_to_int.json")
with open(json_path, "r") as f:
    itemid_to_int = json.load(f)
new_photo_summary["ItemId"] = new_photo_summary["business_id"].map(itemid_to_int)
new_photo_summary.drop(columns=["business_id"], inplace=True)

# Drop unnecessary columns
new_photo_summary = new_photo_summary.drop(['summary', 'keywords', 'indoor_color_tone',
       'venue_style', 'food_style', 'drink_style', 'target_audience',
       'special_features', 'name', 'attributes', 'combined_text',
       'attributes_text', 'photo_text'], axis=1)
# Rename embedding column
new_photo_summary = new_photo_summary.rename(columns={"ada_embedding": "embedding"})
# Ensure ItemId is an integer if it exists
new_photo_summary["ItemId"] = new_photo_summary["ItemId"].astype(int)

# Validate embedding dimensions and types
lengths = new_photo_summary["embedding"].apply(len).unique()
types = new_photo_summary["embedding"].apply(lambda x: type(x)).unique()
print(lengths)  # Ensure all embeddings have the same dimensions
print(types)  # Ensure all embeddings are lists or correct data types

if len(lengths) == 1 and all(t == list for t in types):
    new_photo_summary.to_csv(OUTPUT, index=False, compression="gzip")
    print(f"Embeddings saved to {OUTPUT}")
else:
    print("Error: Inconsistent embedding dimensions or types. File not saved.")