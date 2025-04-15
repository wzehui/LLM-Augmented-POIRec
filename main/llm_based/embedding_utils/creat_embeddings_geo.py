import pandas as pd
import json
import os.path as osp
import h3
from main.llm_based.embedding_utils.openai_utils import set_embeddings_from_df

# 设置工作路径
WORKING_DIR = "../../../yelp"
# OUTPUT = f"{WORKING_DIR}/embeddings/geo_embeddings_openai.csv.gz"
OUTPUT = f"{WORKING_DIR}/embeddings/geo_no_neighbor_embeddings_openai.csv.gz"

# 读取 business 数据
product_lookup = pd.read_csv(
    f"{WORKING_DIR}/csv/business_filtered.csv",
    usecols=['business_id', 'name', 'address', 'city', 'state',
             'postal_code', 'latitude', 'longitude'])

# 1️⃣ 计算 H3 代码（Res 5 - Res 10），并添加相邻六边形
def generate_h3_with_neighbors(lat, lon):
    """ 计算 H3 Res 5 - Res 10，并添加真正的邻近六边形（不包括自身） """
    if pd.isna(lat) or pd.isna(lon):
        return ""
    h3_texts = []
    for res in range(5, 11, 2):
        h3_code = h3.latlng_to_cell(lat, lon, res)
        h3_center = h3.cell_to_latlng(h3_code)  # 获取 H3 的中心点
        h3_center_str = f"({h3_center[0]:.6f}, {h3_center[1]:.6f})"
        neighbors = list(set(h3.grid_disk(h3_code, 1)) - {h3_code})
        neighbor_str = ", ".join(neighbors)
        h3_texts.append(f"H3-{res}: {h3_code}, Center: {h3_center_str}, Neighbors:"
                        f" {neighbor_str}")
    return " | ".join(h3_texts)

product_lookup["h3_codes"] = product_lookup.apply(
    lambda row: generate_h3_with_neighbors(row["latitude"], row["longitude"]), axis=1
)

# 2️⃣ 结构化 Address 信息
# product_lookup["location_info"] = product_lookup.apply(
#     lambda row: (
#         f"Address: {row['address']} | City: {row['city']} | State: {row['state']} | "
#         f"Coordinates: ({row['latitude']:.6f}, {row['longitude']:.6f})"
#     ) if (
#         pd.notna(row['address']) and pd.notna(row['city']) and pd.notna(row['state']) and
#         pd.notna(row['latitude']) and pd.notna(row['longitude'])
#     ) else "", axis=1
# )
product_lookup["location_info"] = product_lookup.apply(
    lambda row: " | ".join(filter(None, [
        f"Address: {row['address']}" if pd.notna(row['address']) else "",
        f"City: {row['city']}" if pd.notna(row['city']) else "",
        f"State: {row['state']}" if pd.notna(row['state']) else "",
        f"Coordinates: ({row['latitude']:.6f}, {row['longitude']:.6f})"
        if pd.notna(row['latitude']) and pd.notna(row['longitude']) else ""
    ])),
    axis=1
)

# 3️⃣ 组合最终的 `combined_location_text`
# product_lookup["combined_text"] = product_lookup.apply(
#     lambda row: f"{row['location_info']} | {row['h3_codes']}".strip(" |"), axis=1
# )
product_lookup["combined_text"] = product_lookup["location_info"]

import tiktoken
# Load OpenAI's tokenizer for "text-embedding-ada-002"
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
# Compute token count for each input
product_lookup['token_count'] = product_lookup['combined_text'].apply(lambda x: len(tokenizer.encode(x)))
# Calculate average token count
average_tokens = product_lookup['token_count'].mean()
print(f"Average token count per input: {average_tokens:.2f}")

product_lookup["combined_text"] = product_lookup["combined_text"].fillna("UNKNOWN LOCATION")
product_lookup["combined_text"] = product_lookup["combined_text"].apply(
    lambda x: "UNKNOWN LOCATION" if x.strip() == "" else x
)
num_unknown_location = (product_lookup["combined_text"] == "UNKNOWN LOCATION").sum()
if num_unknown_location > 0:
    raise ValueError("ERROR: Found {} businesses with 'UNKNOWN LOCATION'. Fix missing location data before proceeding.".format(num_unknown_location))

# # 4️⃣ 计算地理 Embedding
step_size = 1000
batch_geo_summary = product_lookup.iloc[0:step_size]
cur_step = step_size
processed_batches = []
num_batches_processed = 0
while not batch_geo_summary.empty:
    print(f"Currently processing {cur_step} of {len(product_lookup)}")
    batch_geo_summary = set_embeddings_from_df(batch_geo_summary, text_column="combined_text")
    processed_batches.append(batch_geo_summary)
    batch_geo_summary = product_lookup.iloc[cur_step:cur_step + step_size]
    cur_step += step_size

# 合并已处理的 batches
new_geo_summary = pd.concat(processed_batches, copy=False)
new_geo_summary["ada_embedding"] = new_geo_summary["ada_embedding"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else x
)

# 5️⃣ 保存 Embedding
json_path = osp.join(WORKING_DIR, "dataset/itemid_to_int.json")
with open(json_path, "r") as f:
    itemid_to_int = json.load(f)
new_geo_summary["ItemId"] = new_geo_summary["business_id"].map(itemid_to_int)
new_geo_summary.drop(columns=["business_id"], inplace=True)

# Drop unnecessary columns
new_geo_summary = new_geo_summary.drop(['name', 'address', 'city', 'state', 'postal_code',
       'latitude', 'longitude', 'h3_codes', 'location_info', 'combined_text'], axis=1)
# Rename embedding column
new_geo_summary = new_geo_summary.rename(columns={"ada_embedding": "embedding"})
# Ensure ItemId is an integer if it exists
new_geo_summary["ItemId"] = new_geo_summary["ItemId"].astype(int)

# Validate embedding dimensions and types
lengths = new_geo_summary["embedding"].apply(len).unique()
types = new_geo_summary["embedding"].apply(lambda x: type(x)).unique()
print(lengths)  # Ensure all embeddings have the same dimensions
print(types)  # Ensure all embeddings are lists or correct data types

if len(lengths) == 1 and all(t == list for t in types):
    new_geo_summary.to_csv(OUTPUT, index=False, compression="gzip")
    print(f"Embeddings saved to {OUTPUT}")
else:
    print("Error: Inconsistent embedding dimensions or types. File not saved.")