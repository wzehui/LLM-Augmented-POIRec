import pandas as pd
import json
import os

# 定义工作目录
WORKING_DIR = "../../../yelp"

# 定义输入文件列表
input_files = [
    f"{WORKING_DIR}/embeddings/product_embeddings_openai.csv.gz",
    f"{WORKING_DIR}/embeddings/geo_embeddings_openai.csv.gz",
    # f"{WORKING_DIR}/embeddings/photo_summary_embeddings_openai.csv.gz",
    # f"{WORKING_DIR}/embeddings/review_summary_embeddings_deepseek.csv.gz",
]

# 定义输出文件
OUTPUT_FILE = f"{WORKING_DIR}/embeddings/meta+geo_embeddings_openai.csv.gz"

# 读取所有数据并合并
merged_df = None
expected_dim = 1536  # 设定每个 embedding 预期的维度

for file in input_files:
    print(f"Reading {file}...")
    df = pd.read_csv(file, compression='gzip')

    # 解析 embedding 列，将 JSON 字符串转换为列表
    df["embedding"] = df["embedding"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x)

    # 检查 embedding 维度是否符合预期
    sample_embedding = df["embedding"].iloc[0]
    if not isinstance(sample_embedding, list) or len(sample_embedding) != expected_dim:
        raise ValueError(f"Error: {file} 的 embedding 维度不匹配，预期 {expected_dim}，但得到 {len(sample_embedding)}")

    # 生成唯一的 embedding 列名（使用文件名前缀）
    file_prefix = os.path.basename(file).replace(".csv.gz", "")
    df.rename(columns={"embedding": f"embedding_{file_prefix}"}, inplace=True)

    # 合并到主 DataFrame
    if merged_df is None:
        merged_df = df
    else:
        merged_df = pd.merge(merged_df, df, on="ItemId", how="outer")

# 处理 embedding：将所有 embedding_* 列进行正确拼接
embedding_columns = [col for col in merged_df.columns if
                     col.startswith("embedding_")]

# 确保所有 embedding 都是列表，避免字符串拼接
for col in embedding_columns:
    merged_df[col] = merged_df[col].apply(
        lambda x: x if isinstance(x, list) else [])

# 逐列拼接 embedding，确保正确合并
merged_df["embedding"] = merged_df[embedding_columns].apply(
    lambda row: sum(row, []), axis=1)

# 检查最终 embedding 维度
sample_final_embedding = merged_df["embedding"].iloc[0]
expected_final_dim = expected_dim * len(input_files)
if len(sample_final_embedding) != expected_final_dim:
    raise ValueError(f"Error: 合并后 embedding 维度不匹配，预期 {expected_final_dim}，但得到 {len(sample_final_embedding)}")
print("All checks passed. Embeddings are correctly merged.")

# 只保留 ItemId 和合并后的 embedding 列
merged_df = merged_df[["ItemId", "embedding"]]

# 保存到 CSV.GZ 文件
print(f"Saving merged embeddings to {OUTPUT_FILE}...")
merged_df.to_csv(OUTPUT_FILE, index=False, compression='gzip')
print("Done!")
