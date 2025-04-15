import openai
import pandas as pd
import base64
import os
import re
import time
import json
import numpy as np
from tqdm import tqdm

gpu = 3  # Set the current GPU index (0, 1, 2, 3)

# Configure paths
WORKING_DIR = "../../../yelp"
RESULTS_DIR = os.path.join(WORKING_DIR, "embeddings")

# OpenAI API authentication
KEY_LINE = 3  # 1-5
with open("./key.txt", "r") as file:
    lines = file.readlines()
    api_key = lines[KEY_LINE - 1].strip()
client = openai.OpenAI(api_key=api_key)

# Load CSV data
csv_path = os.path.join(WORKING_DIR, "csv/photo_filtered_refined.csv")
photo_df = pd.read_csv(csv_path)

# Load processed business IDs
PROCESSED_IDS_FILE = os.path.join(RESULTS_DIR, f"processed_ids_{gpu}.json")
processed_ids = set()
if os.path.exists(PROCESSED_IDS_FILE):
    with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
        processed_ids = set(json.load(f))

# Load error business IDs
ERROR_IDS_FILE = os.path.join(RESULTS_DIR, f"error_ids_{gpu}.json")
error_ids = set()
if os.path.exists(ERROR_IDS_FILE):
    with open(ERROR_IDS_FILE, "r", encoding="utf-8") as f:
        error_ids = set(json.load(f))

# Process all business IDs
all_business_ids = photo_df["business_id"].unique()
remaining_business_ids = [bid for bid in all_business_ids if bid not in processed_ids and bid not in error_ids]

# Image folder
image_folder = os.path.join(WORKING_DIR, "raw/yelp_photos/photos")

# API batch task settings
BATCH_SIZE = 50  # Each batch processes 50 business IDs
max_batches = len(remaining_business_ids)  # Total number of businesses
start_index = (gpu * max_batches) // 4
end_index = ((gpu + 1) * max_batches) // 4
assigned_ids = remaining_business_ids[start_index:end_index]

# Split into multiple batches
batches = [assigned_ids[i:i + BATCH_SIZE] for i in range(0, len(assigned_ids), BATCH_SIZE)]

# Process each batch
for batch_idx, batch in enumerate(batches):
    print(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} business IDs.")

    batch_requests = []  # Clear batch_requests to avoid duplicate requests

    for business_id in tqdm(batch, desc=f"Batch {batch_idx + 1}/{len(batches)} Processing", unit="business_id"):
        selected_photos = photo_df[photo_df["business_id"] == business_id]

        # Get all image paths
        all_image_paths = [os.path.join(image_folder, f"{photo_id}.jpg") for photo_id in selected_photos["photo_id"]]

        # Filter existing images
        existing_image_paths = [img for img in all_image_paths if os.path.exists(img)]
        missing_count = len(all_image_paths) - len(existing_image_paths)

        # Print number of missing images if some are missing
        if missing_count > 0:
            print(f"⚠️ Business {business_id} is missing {missing_count} image(s).")

        # If all images are missing, mark as error and skip
        if len(existing_image_paths) == 0:
            error_ids.add(business_id)
            print(f"❌ Business {business_id} has no available images. Skipping.")
            continue

        # Encode images as base64
        image_b64_list = []
        for img in existing_image_paths:
            with open(img, "rb") as image_file:
                image_b64_list.append(base64.b64encode(image_file.read()).decode("utf-8"))

        # Construct API prompt
        text_prompt = (
            "Analyze the provided image(s) strictly based on visible elements. "
            "If multiple images are available, synthesize insights holistically, capturing common patterns. "
            "If only one image is provided, analyze it comprehensively without assuming missing context. "
            "Return output in valid JSON format:\n"            
            "{\n"
            '  "summary": "A concise one- or two-sentence overview capturing the overall environment, mood, and atmosphere.",\n'
            '  "keywords": ["Up to 5 most relevant visual keywords reflecting key aspects of the image(s)."],\n'
            '  "themes": {\n'
            '    "Indoor_Color_Tone": ["Summarize the predominant color scheme observed. If no indoor elements are present, return an empty list []."],\n'
            '    "Venue_Style": ["Describe the overall design style considering materials, furniture, and decor."],\n'
            '    "Food_Style": ["Describe the visual characteristics of food images without naming specific items. If no food is present, return an empty list []."],\n'
            '    "Drink_Style": ["Describe the visual attributes of beverages including color, clarity, froth, and garnishes. If no drinks are present, return an empty list []."],\n'
            '    "Target_Audience": ["Identify patterns in customer demographics and dress codes based on visible individuals. If no people are visible, return an empty list []."],\n'
            '    "Special_Features": ["Summarize any distinctive venue features such as outdoor seating, live music, self-checkout, or unique design elements. If no special features are explicitly visible, return an empty list []."]\n'
            '  }\n'
            "}\n\n"
        )

        # Construct API messages
        messages = [
            {"role": "system", "content": "You are an expert in business and restaurant analysis."},
            {"role": "user", "content": text_prompt},
        ]

        # Send images and related metadata
        for i, img_b64 in enumerate(image_b64_list):
            photo_row = selected_photos.iloc[i]  # Get row data
            photo_id = photo_row["photo_id"]
            label = photo_row["label"] if "label" in selected_photos.columns else "Unknown"
            caption = photo_row["caption"] if "caption" in selected_photos.columns else ""

            # Ensure caption is not NaN
            if isinstance(caption, float) and np.isnan(caption):
                caption = ""

            # Construct metadata text
            image_metadata = f"Photo ID: {photo_id}\nLabel: {label}\nCaption: {caption}".strip()

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": image_metadata},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }
            )

        batch_requests.append({
            "custom_id": f"business-{business_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1  # Lower temperature for better stability
            }
        })

    # **Step 3: Generate JSONL file**
    batch_path = os.path.join(RESULTS_DIR, f"batch_request_{gpu}.jsonl")
    with open(batch_path, "w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    print(f"✅ JSONL file created: {batch_path}")

    # **Step 4: Upload JSONL file**
    uploaded_file = client.files.create(
        file=open(batch_path, "rb"),
        purpose="batch"
    )

    file_id = uploaded_file.id
    print(f"✅ File uploaded successfully. File ID: {file_id}")

    # **Step 5: Create batch job**
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    batch_id = batch_response.id
    print(f"✅ Batch job {batch_idx} submitted! Batch ID: {batch_id}")

    # Wait for batch processing
    while True:
        batch_status = client.batches.retrieve(batch_id)
        if batch_status.status in ["completed", "failed", "cancelled", "expired"]:
            break
        print(f"⌛ Waiting for batch {batch_idx} processing... Current status: {batch_status.status}")
        time.sleep(30)

    # Retrieve and parse results
    if batch_status.status == "completed":
        result_file_id = batch_status.output_file_id
        result_file = client.files.content(result_file_id)
        result_text = result_file.read().decode("utf-8").strip()

        # Parse JSONL output
        response_data = [json.loads(line) for line in result_text.split("\n") if
                         line]

        print(response_data)

        # Process each business JSON result
        all_results = {}
        error_entries = []
        for entry in response_data:
            try:
                business_id = entry["custom_id"].replace("business-", "")
                response_text = \
                entry["response"]["body"]["choices"][0]["message"]["content"]

                # ✅ Remove Markdown code block safely using regex
                response_text = re.sub(r"```json\s*", "",
                                       response_text)  # Remove opening ```
                response_text = re.sub(r"\s*```$", "",
                                       response_text)  # Remove closing ```

                # ✅ Check if response_text contains valid JSON before parsing
                try:
                    response_json = json.loads(
                        response_text.strip())  # Attempt to parse
                except json.JSONDecodeError:
                    raise ValueError(
                        f"Invalid JSON format for business {business_id}: {response_text}")

                # Store in results dictionary
                all_results[business_id] = response_json

                # **Immediately save the processed business result**
                batch_results_file = os.path.join(RESULTS_DIR,
                                                  f"visual_summary_"
                                                  f"{batch_idx}_{gpu}.json")
                with open(batch_results_file, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=4, ensure_ascii=False)

                # Mark as processed
                processed_ids.add(business_id)
                with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
                    json.dump(list(processed_ids), f, indent=4,
                              ensure_ascii=False)
                print(
                    f"✅ Results saved for business {business_id} in {batch_results_file}")

            except Exception as e:
                print(f"❌ Error processing business {business_id}: {e}")

                # **Log the failed business ID**
                error_entries.append(
                    {"business_id": business_id, "error": str(e)})

        # **Save all errors in structured JSON format**
        if error_entries:
            with open(ERROR_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(error_entries, f, indent=4, ensure_ascii=False)
            print(f"✅ Errors saved in {ERROR_IDS_FILE}")

print("✅ All batch processing completed!")