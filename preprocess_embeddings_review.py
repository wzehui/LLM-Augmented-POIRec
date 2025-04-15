import os
import json
import pandas as pd

# Specify file paths
json_dir = "./yelp/embeddings/"
missing_output_path = "./yelp/embeddings/missing_json_data.json"  # Path to save problematic JSON data

# Store problematic business_ids
missing_business_ids = set()  # Record all business_ids that require full JSON data to be saved
data_list = []
saved_business_ids = set()

# Iterate through all files in the directory that match batch_results_*.json
for file_name in os.listdir(json_dir):
    if file_name.startswith("batch_results_") and file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        # Read JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)  # Parse JSON
        except json.JSONDecodeError as e:
            print(f"❌ Failed to process: {file_name} - JSON decode error: {e}")
            # **If JSON parsing fails, save the file name**
            missing_business_ids.add(file_name)
            continue  # Skip this file

        # Iterate through business_ids and check for missing fields
        for business_id, info in json_data.items():
            if isinstance(info, dict):
                try:
                    sentiment = info.get("sentiment", {})

                    # Ensure sentiment is a dictionary
                    if not isinstance(sentiment, dict):
                        sentiment = {}

                    # Get sentiment_score and sentiment_confidence
                    sentiment_score = sentiment.get("score", None)
                    sentiment_confidence = sentiment.get("confidence", None)

                    # Get other fields
                    summary = info.get("summary", "").strip()
                    keywords = ", ".join(info.get("keywords", [])).strip()
                    themes_dict = info.get("themes", {})
                    themes = "; ".join(
                        [f"{key}: {', '.join(value)}" for key, value in themes_dict.items()]
                    ).strip()

                    # **If any field is missing, add business_id to lookup list**
                    if (
                        any(v in [None, "", "{}"] for v in [summary, keywords, themes, sentiment_score, sentiment_confidence])
                        or not isinstance(sentiment_confidence, (int, float))
                    ):
                        missing_business_ids.add(business_id)
                    else:
                        data_list.append({
                            "business_id": str(business_id),
                            "sentiment_score": float(sentiment_score) if sentiment_score is not None else None,
                            "sentiment_confidence": float(sentiment_confidence) if sentiment_confidence is not None else None,
                            "summary": str(summary),
                            "keywords": str(keywords),
                            "themes": str(themes)
                        })
                        saved_business_ids.add(business_id)  # Record successfully saved business_id

                except Exception as e:
                    print(f"❌ Failed to parse: {file_name} - {business_id}")
                    missing_business_ids.add(business_id)  # Record failed business_id

# **Second pass: lookup and store full JSON segments for problematic business_ids**
missing_data = {}  # Store full JSON blocks

for file_name in os.listdir(json_dir):
    if file_name.startswith("batch_results_") and file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)  # Parse JSON
        except json.JSONDecodeError:
            continue  # Skip invalid JSON files

        for business_id in missing_business_ids.copy():
            if business_id in saved_business_ids:
                print(f"⚠️ Problematic ID {business_id} was already regenerated")
                missing_business_ids.remove(business_id)
            elif business_id in json_data:
                missing_data[business_id] = json_data[business_id]

# **Save problematic full JSON segments**
with open(missing_output_path, "w", encoding="utf-8") as f:
    json.dump(missing_data, f, ensure_ascii=False, indent=4)

print(f"\n⚠️ Found {len(missing_data)} problematic business entries, saved to {missing_output_path}")

# Convert to DataFrame and save CSV
df = pd.DataFrame(data_list)
df = df.drop_duplicates(subset=['business_id'], keep='first')
df["business_id"] = df["business_id"].astype(str)
df.to_csv(os.path.join(json_dir, "review_summary_deepseek.csv"), index=False, encoding="utf-8-sig")
print(f"📊 Total number of business IDs: {len(df)}")

# Load reference business list for ID comparison
csv_file_path = "/data_private/LLM-Sequential-Recommendation/yelp/csv/business_filtered.csv"
business_df = pd.read_csv(csv_file_path, usecols=["business_id"])
print(len(business_df['business_id'].unique()))

a = set(business_df['business_id'].unique())
b = set(df['business_id'].unique())

# Check if all business_ids in 'a' are present in 'b'
if a.issubset(b):
    print("All business IDs in 'a' are contained in 'b'.")
else:
    missing_ids = a - b  # Compute missing IDs
    print(f"Some business IDs in 'a' are missing from 'b'. Missing count: {len(missing_ids)}")
    print(missing_ids)

# Check for extra business_ids in 'b'
additional_ids = b - a  # Compute extra IDs
print(f"Some business IDs are additional. Additional count: {len(additional_ids)}")
print(additional_ids)