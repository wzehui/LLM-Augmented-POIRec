import os
import numpy as np
import pandas as pd
import datetime
from main.data.session_dataset import SessionDataset
from main.eval.evaluation import Evaluation, EvaluationReport, metrics
from main.transformer.bert.bert_with_embeddings import BERTWithEmbeddings

# Settings
INCLUDE = {
    "LLM2BERT4Rec",
}

DATASET_FILENAME = "../yelp/dataset/dataset.pickle"
EXPERIMENTS_FOLDER = "../results"

CORES = 20
EARLY_STOPPING_PATIENCE = 10
IS_VERBOSE = True
FILTER_PROMPT_ITEMS = True
MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION = 500
PRED_BATCH_SIZE = 1000
PRED_SEEN = False
TRAIN_VAL_FRACTION = 0.1
TOP_Ks = [10, 20]

# Embedding paths
EMBEDDING_PATHS = {
    "product": "../yelp/embeddings/product_embeddings_openai.csv.gz",
    "geo": "../yelp/embeddings/geo_embeddings_openai.csv.gz",
    "review": "../yelp/embeddings/review_summary_embeddings_deepseek.csv.gz",
    "photo": "../yelp/embeddings/photo_summary_embeddings_openai.csv.gz",
    "meta+geo": "../yelp/embeddings/meta+geo_embeddings_openai.csv.gz",
    "UGC": "../yelp/embeddings/UGC_embeddings_openai.csv.gz",
    "all": "../yelp/embeddings/total_embeddings_openai.csv.gz",
}

# Main dataset (no embeddings)
dataset = SessionDataset.from_pickle(DATASET_FILENAME)

# Model mapping
MODEL_CLASS_MAPPING = {
    "LLM2BERT4Rec": BERTWithEmbeddings,
}

# Best configs
BEST_CONFIGS = {
    "LLM2BERT4Rec": {
        "product": [
            {
                "N": 25,
                "L": 9,
                "activation": "relu",
                "drop_rate": 0.2114116576742054,
                "emb_dim": 160,
                "fit_batch_size": 256,
                "h": 8,
                "mask_prob": 0.17981856587582784,
                "optimizer_kwargs": {
                    "clipnorm": 20,
                    "learning_rate": 0.0001116640835093072,
                    "weight_decay": 0.03933356109972018,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.042913764
        ],
        "geo": [
            {
                "N": 25,
                "L": 8,
                "activation": "relu",
                "drop_rate": 0.13094899974328833,
                "emb_dim": 128,
                "fit_batch_size": 96,
                "h": 10,
                "mask_prob": 0.15131412878059927,
                "optimizer_kwargs": {
                    "clipnorm": 45,
                    "learning_rate": 0.00011678733456568815,
                    "weight_decay": 0.021366149486251897,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.040035729
        ],
        "review": [
            {
                "N": 25,
                "L": 7,
                "activation": "relu",
                "drop_rate": 0.2660094647786191,
                "emb_dim": 128,
                "fit_batch_size": 128,
                "h": 5,
                "mask_prob": 0.15392208666254287,
                "optimizer_kwargs": {
                    "clipnorm": 31,
                    "learning_rate": 0.00012960091446054863,
                    "weight_decay": 0.02680706208954032,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.039749222
        ],
        "photo": [
            {
                "N": 25,
                "L": 6,
                "activation": "relu",
                "drop_rate": 0.11836856363372165,
                "emb_dim": 160,
                "fit_batch_size": 192,
                "h": 9,
                "mask_prob": 0.19022687671755362,
                "optimizer_kwargs": {
                    "clipnorm": 18,
                    "learning_rate": 0.00013838294144625875,
                    "weight_decay": 0.024563037487537173,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.041265582
        ],
        "meta+geo": [
            {
                "N": 25,
                "L": 7,
                "activation": "relu",
                "drop_rate": 0.2277061551836269,
                "emb_dim": 640,
                "fit_batch_size": 96,
                "h": 9,
                "mask_prob": 0.28954707396082974,
                "optimizer_kwargs": {
                    "clipnorm": 25,
                    "learning_rate": 1.9203711273019946e-05,
                    "weight_decay": 0.018351706645275447,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.050934037
        ],
        "UGC": [
            {
                "N": 25,
                "L": 7,
                "activation": "relu",
                "drop_rate": 0.37768222820928155,
                "emb_dim": 448,
                "fit_batch_size": 128,
                "h": 4,
                "mask_prob": 0.2206699236527841,
                "optimizer_kwargs": {
                    "clipnorm": 24,
                    "learning_rate": 1.3494996228313363e-05,
                    "weight_decay": 0.032825011282604385,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.0495861150691394
        ],
        "all": [
            {
                "N": 25,
                "L": 7,
                "activation": "relu",
                "drop_rate": 0.32707504167569185,
                "emb_dim": 896,
                "fit_batch_size": 80,
                "h": 5,
                "mask_prob": 0.2651866042945253,
                "optimizer_kwargs": {
                    "clipnorm": 17,
                    "learning_rate": 1.3518300187503806e-05,
                    "weight_decay": 0.043147667567529246,
                },
                "transformer_layer_kwargs": {"layout": "FDRN"},
            },  # Best trial 0.049861501
        ],
    },
}

# Core function to train & predict
def train_and_predict_n(model_class, model_config, dataset, with_item_data,
                        n_trials=5, result_path="results.csv"):
    all_results_list = []
    all_trials_records = []

    for param_set in model_config:
        model_for_info = model_class(**param_set)
        model_name = model_for_info.name()
        trial_results = []
        best_ndcg_20 = -np.inf
        best_trial_result = {}

        for trial_num in range(n_trials):
            model = model_class(**param_set)
            if with_item_data:
                model.train(dataset.get_train_data(), dataset.get_item_data())
            else:
                model.train(dataset.get_train_data())
            model_predictions = model.predict(dataset.get_test_prompts(), top_k=max(TOP_Ks))

            dependencies = {
                metrics.MetricDependency.NUM_ITEMS: dataset.get_unique_item_count(),
                metrics.MetricDependency.ITEM_COUNT: dataset.get_item_counts(),
                metrics.MetricDependency.SAMPLE_COUNT: dataset.get_sample_counts(),
            }

            model_report = None
            for top_k in TOP_Ks:
                report: EvaluationReport = Evaluation.eval(
                    predictions=model_predictions,
                    ground_truths=dataset.get_test_ground_truths(),
                    model_name=model_name,
                    top_k=top_k,
                    metrics_per_sample=False,
                    dependencies=dependencies,
                    cores=1,
                )
                if model_report is None:
                    model_report = report
                else:
                    model_report.results.update(report.results)

            trial_results.append(model_report.results)

            trial_record = {
                "trial_id": trial_num + 1,
                "Model Name": model_name,
            }
            trial_record.update(model_report.results)
            all_trials_records.append(trial_record)

            if model_report.results['NDCG@20'] > best_ndcg_20:
                best_ndcg_20 = model_report.results['NDCG@20']
                best_trial_result = model_report.results.copy()

        metrics_summary = {}
        for metric in trial_results[0]:
            metric_values = [trial[metric] for trial in trial_results]
            metrics_summary[f'{metric}_mean'] = np.mean(metric_values)
            metrics_summary[f'{metric}_std'] = np.std(metric_values)

        metrics_summary['Best_NDCG@20'] = best_ndcg_20
        param_set_filtered = {k: v for k, v in param_set.items() if k not in ['cores', 'is_verbose']}

        result_entry = {
            "Model Name": model_name,
            "embedding_type": param_set.get("embedding_type", "N/A"),
            **param_set_filtered,
            **{f"Best_{k}": best_trial_result[k] for k in sorted(best_trial_result)},
            **{k: metrics_summary[k] for k in sorted(metrics_summary)}
        }
        all_results_list.append(result_entry)

    trial_result_path = result_path.replace(".csv", "_trials.csv")
    pd.DataFrame(all_trials_records).to_csv(trial_result_path, index=False)

    results_df = pd.DataFrame(all_results_list)
    metric_order = [col for metric in ['NDCG@10', 'HitRate@10', 'MRR@10', 'Catalog coverage@10', 'Serendipity@10', 'Novelty@10',
                                       'NDCG@20', 'HitRate@20', 'MRR@20', 'Catalog coverage@20', 'Serendipity@20', 'Novelty@20']
                    for col in [f'Best_{metric}', f'{metric}_mean', f'{metric}_std'] if col in results_df.columns]
    columns_order = ['Model Name', 'embedding_type'] + [col for col in results_df.columns if col not in ['Model Name', 'embedding_type', 'cores', 'is_verbose'] + metric_order] + metric_order

    results_df = results_df[columns_order].drop(columns=['cores', 'is_verbose'], errors='ignore')
    results_df.sort_values(by="NDCG@20_mean", ascending=False, inplace=True)

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    if os.path.exists(result_path):
        old_results = pd.read_csv(result_path)
        results_df = pd.concat([old_results, results_df]).drop_duplicates(subset=["Model Name", "embedding_type"], keep="last").reset_index(drop=True)

    results_df.to_csv(result_path, index=False)

# Unified execution loop
def run_all_experiments():
    for model_name in INCLUDE:
        if model_name not in BEST_CONFIGS:
            continue
        model_class = MODEL_CLASS_MAPPING[model_name]
        with_item_data = model_name in {"SKNN_EMB", "LLMSeqSim"}

        for emb_type in EMBEDDING_PATHS:
            if emb_type not in BEST_CONFIGS[model_name]:
                continue

            dataset_used = dataset
            emb_path = EMBEDDING_PATHS[emb_type]
            config_list = BEST_CONFIGS[model_name][emb_type]
            enriched_configs = []
            for config in config_list:
                cfg = config.copy()
                if with_item_data:
                    cfg.update({
                        "sample_random_state": 2025,
                        "cores": CORES,
                        "is_verbose": IS_VERBOSE,
                        "filter_prompt_items": FILTER_PROMPT_ITEMS,
                        "max_session_length_for_decay_precomputation": MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION,
                    })
                else:
                    cfg.update({
                        "product_embeddings_location": emb_path,
                        "cores": CORES,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "is_verbose": IS_VERBOSE,
                        "pred_batch_size": PRED_BATCH_SIZE,
                        "pred_seen": PRED_SEEN,
                        "train_val_fraction": TRAIN_VAL_FRACTION,
                        "red_method": "PCA",
                        "red_params": {},
                    })
                enriched_configs.append(cfg)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            result_file = os.path.join(EXPERIMENTS_FOLDER, f"results_{model_name}_{emb_type}_{timestamp}.csv")
            train_and_predict_n(model_class, enriched_configs, dataset_used, with_item_data, result_path=result_file)

# Run experiments
run_all_experiments()
