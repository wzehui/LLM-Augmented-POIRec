import os
import pandas as pd
import pickle
from main.data.session_dataset import SessionDataset
from main.eval.evaluation import Evaluation, EvaluationReport, metrics
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from main.transformer.sasrec.sasrec import SASRec
from main.transformer.bert.bert import BERT
from main.grurec.grurec import GRURec
from main.sknn.sknn import SessionBasedCF

import optuna
from optuna.samplers import TPESampler

INCLUDE = {
    # "GRU4Rec",
    # "SASRec",
    "BERT4Rec",
    # "V-SKNN",
}

DATASET_FILENAME_TRAIN = "../yelp/dataset/dataset_training.pickle"
EXPERIMENTS_FOLDER = "../results"

if not os.path.exists(EXPERIMENTS_FOLDER):
    os.makedirs(EXPERIMENTS_FOLDER)

# Model configuration
CORES = 20
EARLY_STOPPING_PATIENCE = 10
IS_VERBOSE = True
FILTER_PROMPT_ITEMS = True
MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION = 400
PRED_BATCH_SIZE = 1000
PRED_SEEN = False
TRAIN_VAL_FRACTION = 0.1
TOP_Ks = [10, 20]
RANDOM_SEED = 2025

dataset_train = SessionDataset.from_pickle(DATASET_FILENAME_TRAIN)

def objective(trial):
    if model_class.__name__ == "GRURec":
        grurec_config = {
            "N": 25,
            "activation": "relu",
            "emb_dim": trial.suggest_int("emb_dim", 128, 512, step=32),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 32, 256, step=32),
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 1024, step=32),
            "optimizer_kwargs": {
                "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.005, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            },
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
        }
        model = model_class(**grurec_config)

    elif model_class.__name__ == "SASRec":
        sasrec_config = {
            "N": 25,
            "L": trial.suggest_int("L", 5, 10),
            "activation": "relu",
            "drop_rate": trial.suggest_float("drop_rate", 0.1, 0.5),
            "emb_dim": trial.suggest_int("emb_dim", 128, 512, step=32),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 32, 256, step=32),
            "h": trial.suggest_int("h", 4, 10),
            "optimizer_kwargs": {
                "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.005, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
            },
            "transformer_layer_kwargs": {"layout": "NFDR"},
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
        }
        model = model_class(**sasrec_config)

    elif model_class.__name__ == "BERT":
        bert_config = {
            "N": 25,
            "L": trial.suggest_int("L", 5, 10),
            "activation": "relu",
            "drop_rate": trial.suggest_float("drop_rate", 0.1, 0.3),
            "emb_dim": trial.suggest_int("emb_dim", 128, 512, step=32),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 32, 256, step=32),
            "h": trial.suggest_int("h", 4, 10),
            "mask_prob": trial.suggest_float("mask_prob", 0.1, 0.2),
            "optimizer_kwargs": {
                "clipnorm": trial.suggest_int("clipnorm", 10, 50),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.001, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.05),
            },
            "transformer_layer_kwargs": {"layout": "FDRN"},  # Fixed
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
        }
        model = model_class(**bert_config)

    elif model_class.__name__ == "SessionBasedCF":
        sample_size = trial.suggest_int("sample_size", 900, 5000)
        similarity_measure = trial.suggest_categorical(
            "similarity_measure", ["dot", "cosine", "jaccard"]
        )
        vsknn_config = {
            "sample_size": sample_size,
            "k": trial.suggest_int("k", 10, sample_size - 1),
            "sampling": trial.suggest_categorical(
                "sampling", ["random", "recent", "idf", "idf_greedy"]
            ),
            "similarity_measure": similarity_measure,
            "idf_weighting": trial.suggest_categorical("idf_weighting",
                                                       [True, False]),
            "is_verbose": IS_VERBOSE,
            "cores": CORES,
            "filter_prompt_items": FILTER_PROMPT_ITEMS,
            "max_session_length_for_decay_precomputation": MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION,
        }
        if similarity_measure != "jaccard":
            vsknn_config["decay"] = trial.suggest_categorical(
                "decay", ["linear", "log", "harmonic", "quadratic"]
            )
        model = model_class(**vsknn_config)

    model.train(dataset_train.get_train_data())

    predictions = model.predict(dataset_train.get_test_prompts(),
                                top_k=min(TOP_Ks))
    report = Evaluation.eval(
        predictions=predictions,
        ground_truths=dataset_train.get_test_ground_truths(),
        model_name=model.name(),
        top_k=min(TOP_Ks),
        metrics=[NormalizedDiscountedCumulativeGain()],
        dependencies={
            metrics.MetricDependency.NUM_ITEMS: dataset_train.get_unique_item_count(),
        },
        metrics_per_sample=False,
    )
    trial_df = report.to_df()
    ndcg = trial_df.at[model.name(), "NDCG@10"]

    return ndcg

model_classes = []
if "GRU4Rec" in INCLUDE:
    model_classes.append(GRURec)
if "SASRec" in INCLUDE:
    model_classes.append(SASRec)
if "BERT4Rec" in INCLUDE:
    model_classes.append(BERT)
if "V-SKNN" in INCLUDE or "SKNN_EMB" in INCLUDE:
    model_classes.append(SessionBasedCF)

for model_class in model_classes:
    OPTUNA_STUDY_FILE =  os.path.join(EXPERIMENTS_FOLDER, f"optuna_study_{model_class.__name__}.pkl")
    SAVE_INTERVAL = 5  # Save progress every 5 trials

    # If the Optuna study file exists, delete it first
    if os.path.exists(OPTUNA_STUDY_FILE):
        os.remove(OPTUNA_STUDY_FILE)
        print("Deleted old Optuna study file.")
    # Create a new Optuna study
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED))
    print("Created new Optuna study.")

    # Define a callback function to save progress periodically
    def save_study_callback(study, trial):
        """Save the Optuna study progress every `SAVE_INTERVAL` trials."""
        if trial.number % SAVE_INTERVAL == 0:
            with open(OPTUNA_STUDY_FILE, "wb") as f:
                pickle.dump(study, f)
            print(f"Saved Optuna study progress at trial {trial.number}.")

    # Run Optuna optimization with automatic saving
    study.optimize(objective, n_trials=50, callbacks=[save_study_callback])

    # Final save after all trials complete
    with open(OPTUNA_STUDY_FILE, "wb") as f:
        pickle.dump(study, f)
    print("Final Optuna study progress saved.")

    sorted_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)
    print("Best trial:")
    print(f"Rank 1 - Value: {sorted_trials[0].value}")
    print(f"Params: {sorted_trials[0].params}\n")
    if len(sorted_trials) > 1:
        print("Second best trial:")
        print(f"Rank 2 - Value: {sorted_trials[1].value}")
        print(f"Params: {sorted_trials[1].params}\n")
    if len(sorted_trials) > 2:
        print("Third best trial:")
        print(f"Rank 3 - Value: {sorted_trials[2].value}")
        print(f"Params: {sorted_trials[2].params}")

    best_trials_data = []
    for rank, trial in enumerate(sorted_trials[:3], start=1):
        best_trials_data.append({
            "Rank": rank,
            "Value": trial.value,
            "Params": trial.params,
        })

    import datetime
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(timestamp)
    # Generate a filename with timestamp
    csv_file_path = os.path.join(
        EXPERIMENTS_FOLDER,
        f"{model_class.__name__}_best_3_{timestamp}.csv"
    )
    df = pd.DataFrame(best_trials_data)
    df["Params"] = df["Params"].apply(lambda x: str(x))
    df.to_csv(csv_file_path, index=False)
    print(f"Best 3 trials saved to {csv_file_path}")

    # Remove the Optuna study file after printing the results
    if os.path.exists(OPTUNA_STUDY_FILE):
        os.remove(OPTUNA_STUDY_FILE)
        print("Optuna study file has been deleted.")