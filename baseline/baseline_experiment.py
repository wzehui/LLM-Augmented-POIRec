import os
import numpy as np
import pandas as pd
from main.data.session_dataset import SessionDataset
from main.eval.evaluation import Evaluation, EvaluationReport, metrics
from main.transformer.bert.bert import BERT


INCLUDE = {
    "BERT4Rec",
}

DATASET_FILENAME = "../yelp/dataset/dataset.pickle"
EXPERIMENTS_FOLDER = "../results"
RESULTS = "../results/results.csv"

# Model configuration
CORES = 20
EARLY_STOPPING_PATIENCE = 10
IS_VERBOSE = True
FILTER_PROMPT_ITEMS = True
MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION = 500
PRED_BATCH_SIZE = 1000
PRED_SEEN = False
TRAIN_VAL_FRACTION = 0.1
TOP_Ks = [10, 20]

dataset = SessionDataset.from_pickle(DATASET_FILENAME)

def train_and_predict_n(model_class, model_config, dataset, with_item_data,
                        n_trials=5):
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
            **param_set_filtered,
            **{f"Best_{k}": best_trial_result[k] for k in sorted(best_trial_result)},
            **{k: metrics_summary[k] for k in sorted(metrics_summary)}
        }

        all_results_list.append(result_entry)

    trial_result_path = RESULTS.replace(".csv", "_trials.csv")
    pd.DataFrame(all_trials_records).to_csv(trial_result_path, index=False)

    results_df = pd.DataFrame(all_results_list)

    metric_order = [col for metric in ['NDCG@10', 'HitRate@10', 'MRR@10', 'Catalog coverage@10', 'Serendipity@10', 'Novelty@10',
                                       'NDCG@20', 'HitRate@20', 'MRR@20', 'Catalog coverage@20', 'Serendipity@20', 'Novelty@20']
                    for col in [f'Best_{metric}', f'{metric}_mean', f'{metric}_std'] if col in results_df.columns]

    columns_order = ['Model Name'] + [col for col in results_df.columns if col not in ['Model Name', 'cores', 'is_verbose'] + metric_order] + metric_order

    results_df = results_df[columns_order].drop(columns=['cores', 'is_verbose'], errors='ignore')
    results_df.sort_values(by="NDCG@20_mean", ascending=False, inplace=True)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    if os.path.exists(RESULTS):
        old_results = pd.read_csv(RESULTS)
        results_df = pd.concat([old_results, results_df]).drop_duplicates(subset=["Model Name"], keep="last").reset_index(drop=True)

    results_df.to_csv(RESULTS, index=False)

if "BERT4Rec" in INCLUDE:
    bert_config = [
        {
            "L": 5,
            "N": 25,
            "activation": "relu",
            "drop_rate": 0.12026337771830388,
            "emb_dim": 224,
            "fit_batch_size": 192,
            "h": 8,
            "mask_prob": 0.16555188310617586,
            "optimizer_kwargs": {
                "clipnorm": 34,
                "learning_rate": 0.00011187793596533849,
                "weight_decay": 0.011377488791816649,
            },
            "transformer_layer_kwargs": {"layout": "FDRN"},
        }
    ]
    for config in bert_config:
        config.update(
            {
                "cores": CORES,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "is_verbose": IS_VERBOSE,
                "pred_batch_size": PRED_BATCH_SIZE,
                "pred_seen": PRED_SEEN,
                "train_val_fraction": TRAIN_VAL_FRACTION,
            }
        )
    train_and_predict_n(BERT, bert_config, dataset, False)