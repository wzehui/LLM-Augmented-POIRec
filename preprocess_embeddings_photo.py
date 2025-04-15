import os
import json
import pandas as pd

# Specify the path to the JSON files
json_dir = "./yelp/embeddings/"
output_path = "./yelp/embeddings/photo_summary_openai.json"  # Path to save the parsed data

data_list = []

# Iterate over all files in the directory that start with visual_summary_ and end with .json
for file_name in os.listdir(json_dir):
    if file_name.startswith("visual_summary_") and file_name.endswith(".json"):
        file_path = os.path.join(json_dir, file_name)

        # Read the JSON file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)  # Parse JSON
        except json.JSONDecodeError as e:
            print(f"❌ Failed to process: {file_name} - JSON decode error: {e}")
            continue  # Skip this file

        # Iterate over business_id entries and extract fields
        for business_id, info in json_data.items():
            if isinstance(info, dict):
                summary = info.get("summary", "").strip()
                keywords = ", ".join(info.get("keywords", [])).strip()

                # Process the themes field
                themes_dict = info.get("themes", {})
                indoor_color_tone = ", ".join(themes_dict.get("Indoor_Color_Tone", []))
                venue_style = ", ".join(themes_dict.get("Venue_Style", []))
                food_style = ", ".join(themes_dict.get("Food_Style", []))
                drink_style = ", ".join(themes_dict.get("Drink_Style", []))
                target_audience = ", ".join(themes_dict.get("Target_Audience", []))
                special_features = ", ".join(themes_dict.get("Special_Features", []))

                # Store all fields, even if empty
                row_data = {
                    "business_id": business_id,
                    "summary": summary,
                    "keywords": keywords,
                    "indoor_color_tone": indoor_color_tone,
                    "venue_style": venue_style,
                    "food_style": food_style,
                    "drink_style": drink_style,
                    "target_audience": target_audience,
                    "special_features": special_features
                }

                # Remove fields with "N/A" values
                filtered_data = {k: v for k, v in row_data.items() if v and v != "N/A"}

                # Only add to data_list if there's at least business_id and one other valid field
                if len(filtered_data) > 1:
                    data_list.append(row_data)

# Convert to DataFrame and export to CSV
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(json_dir, "photo_summary_openai.csv"), index=False, encoding="utf-8-sig")
print(f"📊 Total number of business IDs: {len(df)}")

# Load reference CSV to compare business_id coverage
photo_summary = pd.read_csv("./yelp/csv/photo_filtered_refined_f_removed.csv")
print(len(photo_summary['business_id'].unique()))

a = set(photo_summary['business_id'].unique())
b = set(df['business_id'].unique())

# Check if all business_ids in photo_summary exist in the generated summary
if a.issubset(b):
    print("All business IDs in 'a' are contained in 'b'.")
else:
    missing_ids = a - b  # Find missing IDs
    print(f"Some business IDs in 'a' are missing from 'b'. Missing count: {len(missing_ids)}")
    print(missing_ids)

# Check for additional business_ids in b not found in a
additional_ids = b - a
print(f"Some business IDs are additional. Additional count: {len(additional_ids)}")
print(additional_ids)

# Uncomment the following to inspect or save the filtered results
# missing_photo_counts = photo_summary[photo_summary['business_id'].isin(missing_ids)]
# print(f"Total missing photos: {len(missing_photo_counts)}")
# filtered_photo_summary = photo_summary[~photo_summary['business_id'].isin(missing_ids)]
# filtered_photo_summary.to_csv(
#     './yelp/csv/photo_filtered_refined_f_removed.csv', index=False, encoding="utf-8")