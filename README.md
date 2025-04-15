# LLM-Augmented Multimodal Embeddings for POI Recommendation
We explore how different types of business-side information—including 
structured metadata, geolocation, user reviews, and user-uploaded 
photos—impact next-POI recommendation. Our pipeline uses LLMs to summarize 
unstructured reviews and photos into structured text, followed by 
standardized embedding generation. Embedding configurations are evaluated 
under various feature combinations using BERT4Rec.  

This repository is built upon the codebase of previous work 
[LLM-Sequential-Recommendation](https://github.com/dh-r/LLM-Sequential-Recommendation.git).

Please cite as follows:
> ```
> ```

---

## Dataset

All experiments are based on the [Yelp Open Dataset](https://www.yelp.com/dataset), extended with multimodal summaries:

| File                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `business.csv`         | Metadata (e.g., name, category, address, coordinates)                      |
| `checkin.csv`          | User-business interaction sequences                                         |
| `review.csv`           | Full user reviews                                                           |
| `photo.csv`            | Yelp photo metadata (labels, image IDs)                                     |
| `review_summary.csv`   | DeepSeek-R1 generated summaries of reviews                                  |
| `photo_summary.csv`    | GPT-4o generated summaries of business images                               |

✅ Available on HuggingFace: [wzehui/Yelp-Multimodal-Recommendation](https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation)  

---

## Environment Setup

**Base Image**  
We recommend using Docker (base image):

```
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
```

**Python version**: `3.11`

If you're setting up locally:

```
# Install Python 3.11 (Ubuntu)
apt update -y
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update -y
apt install -y python3.11 python3.11-venv python3.11-distutils
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --config python3
```

---

**Poetry Installation**

We recommend using [`pipx`](https://pypa.github.io/pipx/) to install [Poetry](https://python-poetry.org):

```
# Install pipx (if not yet installed)
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade --force-reinstall pip
python3 -m pip install --user pipx
python3 -m pipx ensurepath
source ~/.bashrc

# Install poetry
pipx install poetry
```

---

**Install Dependencies & Activate**

```
poetry install
```

Now you can activate the environment:

```
source $(poetry env info --path)/bin/activate
```

---

## Data Processing

The data processing pipeline is based on the [Yelp Open Dataset](https://www.yelp.com/dataset) and consists of two main steps:

### (1) Preprocessing
Run `preprocessing.py` to convert Yelp’s original JSON files (e.g., `business.json`, `review.json`) into CSV format. The resulting files will be saved under `./yelp/csv/`.

### (2) Filtering
Run `filtering.py` to clean and filter the dataset for sequential POI recommendation. The main steps include:

- **Time Span Filtering**: Keep check-in data between 2017 and 2019 to ensure temporal consistency and avoid COVID-related bias.
- **Bot User Removal**: Identify and exclude users with unrealistically fast travel speeds between locations.
- **Minimum Interaction Filtering**: Apply a 10-core filter to retain users and businesses with at least 10 check-ins.
- **Friend Network Cleaning**: Update user friendship information to include only valid users retained after filtering.

The processed and filtered data will be saved to `./yelp/csv/`.

**💡 Alternative:** if you do not wish to run the preprocessing locally, the 
preprocessed and filtered version is available on HuggingFace: **[wzehui/Yelp-Multimodal-Recommendation](https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation)**
. Please place the unzipped dataset under: `./yelp/csv/`

### (3) Dataset Splitting
Run `preparation_dataset.py` to remap original user and business IDs to integer indices, and split the check-in data into training, validation, and test sets.
The resulting files will be saved under: `./yelp/dataset`

**Example output files:**
- `dataset_training`: training set (train/validation) used during hyperparameter tuning.
- `dataset`: full train/validation/test set used for final evaluation.

---

## LLM-Based Summarization
This step generates structured summaries from unstructured content using large language models (LLMs). Specifically:

- **Review Summarization:** Run `main/llm_based/embedding_utils/review_summary_deepseek.py` to summarize user reviews for each business using DeepSeek-R1
- **Photo Summarization:** Run `main/llm_based/embedding_utils/photo_summary_openai.py` to summarize visual 
  content using OpenAI’s GPT-4o

These scripts aggregate all available reviews and photos for each business and generate concise, structured outputs (e.g., summaries, keywords, sentiment, visual themes).  
Then, use `preprocess_embeddings_review.py` and `preprocess_embeddings_photo.
py` to processes review and photo summaries and generates `review_summary.
csv` and `photo_summary.csv` respectively.

**💡 Alternative:** summary files are available on HuggingFace: **[wzehui/Yelp-Multimodal-Recommendation (https://huggingface.co/datasets/wzehui/Yelp-Multimodal-Recommendation)**. Please place the unzipped dataset under: `./yelp/csv/`

---

## Embedding Generation
The following Python scripts are used to generate various types of embeddings:
- `create_embeddings.py`: Metadata feature category  
- `create_embeddings_geo.py`: Geolocation feature category  
- `create_embeddings_review.py`: User Feedback feature category  
- `create_embeddings_photo.py`: Business Attributes feature category  
- `create_embeddings_meta+geo.py`: Metadata + Geolocation combination  
- `create_embeddings_UGC.py`: User Feedback + Business Attributes combination  
- `create_embeddings_total.py`: All feature combination  

To generate embeddings, run the appropriate script depending on the desired input modality. The output embeddings will be saved in the designated `./yelp/embeddings/` directory.

---

## Model Training
These scripts are used to search for optimal hyperparameters for different 
feature categories and their combinations in our experiments.

- `baseline_experiment_tuning_no.py`: Tuning for no feature input (trajectory 
  only baseline)
- `baseline_experiment_tuning_single.py`: Tuning for each single feature category independently
- `baseline_experiment_tuning_due.py`: Tuning for all pairwise combinations (e.
  g., meta+geo, feedback+attribute)
- `baseline_experiment_tuning_all.py`: Tuning for full combination

All best-performing hyperparameter combinations are saved under the `results/` directory.

---

## Model Test
These scripts are used to evaluate model performance on the test dataset 
using the best hyperparameter configurations identified during the tuning phase.

- `baseline_experiment.py`: Evaluate trajectory-only models (without any 
  side information)
- `baseline_experiment_modality.py`: Evaluate item augmented with side 
  information (modality-aware)

The NDCG, Hit Rate, MRR, Coverage, Serendipity, and Novelty results generated from 5 trials are saved in the `results/` directory.
