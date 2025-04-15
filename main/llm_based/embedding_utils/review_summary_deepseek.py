import pandas as pd
import json
import os
import re
import requests
from tqdm import tqdm

# Configuration parameters
WORKING_DIR = "../../../yelp"
RESULTS_DIR = "../../../yelp/embeddings/"
PROCESSED_IDS_FILE = os.path.join(RESULTS_DIR, "processed_ids_2.json")
ERROR_LOG_FILE = os.path.join(RESULTS_DIR, "error_ids_2.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load already processed business_id
if os.path.exists(PROCESSED_IDS_FILE):
    with open(PROCESSED_IDS_FILE, "r", encoding="utf-8") as f:
        processed_ids = set(json.load(f))
else:
    processed_ids = set()

# Load error-logged business_id
if os.path.exists(ERROR_LOG_FILE):
    with open(ERROR_LOG_FILE, "r", encoding="utf-8") as f:
        error_log = json.load(f)
        error_ids = {entry["business_id"] for entry in error_log}
else:
    error_log = []
    error_ids = set()

# Load review data
product_lookup = pd.read_csv(
    f"{WORKING_DIR}/csv/review_filtered.csv",
    usecols=['review_id', 'user_id', 'business_id', 'text', 'stars', 'reactions']
)

# Get all unique business_ids
all_business_ids = product_lookup['business_id'].unique().tolist()

BATCH_SIZE = 100
gpu_id = 0  # GPU ID (e.g., 0 / 1 / 2)
total_businesses = len(all_business_ids)
max_batches = (total_businesses + BATCH_SIZE - 1) // BATCH_SIZE
start_index = (2 * max_batches) // 3
end_index = ((2 + 1) * max_batches) // 3
mid_index = (start_index + end_index) // 2
num_batches = mid_index - start_index
print(f"🚀 GPU {gpu_id}: Processing from batch {start_index} to {mid_index} ({num_batches} batches)")

# Step 1: Process each business_id and generate summaries
for batch_idx, i in enumerate(range(start_index, mid_index), start=start_index):
    batch_ids = all_business_ids[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    batch_results_file = os.path.join(RESULTS_DIR, f"batch_results_{batch_idx}.json")

    # Check if batch file already exists
    if os.path.exists(batch_results_file):
        with open(batch_results_file, "r", encoding="utf-8") as f:
            try:
                batch_results = json.load(f)
            except json.JSONDecodeError:
                batch_results = {}  # Possibly corrupted, reinitialize
    else:
        batch_results = {}

    print(f"🚀 Processing batch {batch_idx + 1}/{mid_index} on GPU {gpu_id}...")

    for business_id in tqdm(batch_ids, desc=f"GPU {gpu_id} Batch {batch_idx + 1}"):
        if business_id in processed_ids or business_id in error_ids:
            print(f"✅ Business {business_id} already processed or in error list. Skipping...")
            continue

        # Get all reviews for the business
        sample_reviews = product_lookup[product_lookup['business_id'] == business_id]['text'].tolist()
        combined_reviews = "\n".join(sample_reviews)

        # Construct request prompt
        prompt = f"""
        The following are customer reviews for a business:
        {combined_reviews}

        Please analyze the reviews and return the output in the following JSON format:

        {{
          "business_id": "{business_id}",
          "summary": "A brief summary of the reviews in one or two sentences.",
          "keywords": ["Top 5 most relevant keywords."],
          "sentiment": {{
            "score": Sentiment score between -1 (very negative) to 1 (very positive),
            "confidence": Confidence score as a decimal number between 0 and 1.
          }},
          "themes": {{
            "Theme1": ["Relevant keywords for this theme."],
            "...": ["Additional themes if present."]
          }}
        }}
        """

        # Call Ollama API (via requests)
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "deepseek-r1:70b",
            "max_tokens": 2048,
            "temperature": 0.3,
            "stream": False,
            "prompt": prompt
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            response_json = response.json()
            result_text = response_json.get("response", "").strip()

            # Remove ```json and ``` code block
            result_text = re.sub(r"```json|```", "", result_text).strip()

            # Remove <think>...</think> content
            result_text = re.sub(r"<think>.*?</think>", "", result_text, flags=re.DOTALL).strip()

            # Check if empty
            if not result_text:
                raise ValueError("Empty response text")

            # Parse JSON
            parsed_result = json.loads(result_text)

            # Save result to batch_results file
            batch_results[business_id] = parsed_result
            with open(batch_results_file, "w", encoding="utf-8") as f:
                json.dump(batch_results, f, indent=4, ensure_ascii=False)

            # Update processed business_id
            processed_ids.add(business_id)
            with open(PROCESSED_IDS_FILE, "w", encoding="utf-8") as f:
                json.dump(list(processed_ids), f, indent=4, ensure_ascii=False)

            print(f"✅ Saved result for business {business_id}")

        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            response_text = "No response"
            if "response" in locals():
                try:
                    data = response.json()
                    response_text = data.get("response", "No response")
                except (json.JSONDecodeError, ValueError):
                    response_text = "No response"

            error_entry = {
                "business_id": business_id,
                "error_message": str(e),
                "response_text": response_text
            }
            error_log.append(error_entry)
            error_ids.add(business_id)

            # Save to error_ids.json
            with open(ERROR_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(error_log, f, indent=4, ensure_ascii=False)

            print(f"❌ Error processing business {business_id}: {e}")