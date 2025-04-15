import os
import pickle
import datetime
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from main.data.session_dataset import SessionDataset
from main.eval.evaluation import Evaluation, EvaluationReport, metrics
from main.eval.metrics.ndcg import NormalizedDiscountedCumulativeGain
from main.llm_based.similarity_model.llm_seq_sim import LLMSeqSim
from main.sknn.sknn import SessionBasedCF
from main.transformer.sasrec.sasrec_with_embeddings import SASRecWithEmbeddings
from main.grurec.grurec_with_embeddings import GRURecWithEmbeddings
from main.transformer.bert.bert_with_embeddings import BERTWithEmbeddings
from main.utils.config_util import extract_config

INCLUDE = {
    # "LLM2GRURec",
    # "LLM2SASRec",
    "LLM2BERT4Rec",
    # "SKNN_EMB",
    # "LLMSeqSim",
}

DATASET_FILENAME = "../yelp/dataset/dataset_training.pickle"

EMBEDDING_PATHS = {
    "all": "../yelp/embeddings/total_embeddings_openai.csv.gz",
}

DATASET_PATHS = {
    "all": "../yelp/embeddings/openai_augmented_dataset_training_t.pickle",
}

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

model_classes = []
if "LLM2GRURec" in INCLUDE:
    model_classes.append(GRURecWithEmbeddings)
if "LLM2SASRec" in INCLUDE:
    model_classes.append(SASRecWithEmbeddings)
if "LLM2BERT4Rec" in INCLUDE:
    model_classes.append(BERTWithEmbeddings)
if "SKNN_EMB" in INCLUDE:
    model_classes.append(SessionBasedCF)
if "LLMSeqSim" in INCLUDE:
    model_classes.append(LLMSeqSim)

def objective(trial):
    if model_class.__name__ == "GRURecWithEmbeddings":
        llm2grurec_config = {
            "N": 25,
            "activation": "relu",
            "emb_dim": trial.suggest_int("emb_dim", 512, 2048, step=64),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 16, 128, step=16),
            "hidden_dim": trial.suggest_int("hidden_dim", 512, 2048, step=64),
            "optimizer_kwargs": {
                "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.002, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.02, 0.2),
            },
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OPENAI_EMBEDDINGS_PATH,
            "red_method": "PCA",
            "red_params": {},
        }
        model = model_class(**llm2grurec_config)

    elif model_class.__name__ == "SASRecWithEmbeddings":
        llm2sasrec_config = {
            "N": 25,
            "L": trial.suggest_int("L", 4, 12),
            "activation": "relu",
            "drop_rate": trial.suggest_float("drop_rate", 0.2, 0.6),
            "emb_dim": trial.suggest_int("emb_dim", 512, 2048, step=64),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 16, 128, step=16),
            "h": trial.suggest_int("h", 4, 12),
            "optimizer_kwargs": {
                "learning_rate": trial.suggest_float("learning_rate", 0.00005, 0.002, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.02, 0.2),
            },
            "transformer_layer_kwargs": {"layout": "NFDR"},
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OPENAI_EMBEDDINGS_PATH,
            "red_method": "PCA",
            "red_params": {},
        }
        model = model_class(**llm2sasrec_config)

    elif model_class.__name__ == "BERTWithEmbeddings":
        llm2bert_config = {
            "N": 25,
            "L": trial.suggest_int("L", 4, 8),
            "activation": "relu",
            "drop_rate": trial.suggest_float("drop_rate", 0.2, 0.5),
            "emb_dim": trial.suggest_int("emb_dim", 512, 2048, step=64),
            "fit_batch_size": trial.suggest_int("fit_batch_size", 16, 128, step=16),
            "h": trial.suggest_int("h", 4, 10),
            "mask_prob": trial.suggest_float("mask_prob", 0.15, 0.3),
            "optimizer_kwargs": {
                "clipnorm": trial.suggest_int("clipnorm", 10, 50),
                "learning_rate": trial.suggest_float("learning_rate",
                                                     0.000005, 0.00005,
                                                     log=True),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.05),
            },
            "transformer_layer_kwargs": {"layout": "FDRN"},
            "cores": CORES,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "is_verbose": IS_VERBOSE,
            "pred_batch_size": PRED_BATCH_SIZE,
            "pred_seen": PRED_SEEN,
            "train_val_fraction": TRAIN_VAL_FRACTION,
            "product_embeddings_location": OPENAI_EMBEDDINGS_PATH,
            "red_method": "PCA",
            "red_params": {},
        }

        model = model_class(**llm2bert_config)

    elif model_class.__name__ == "SessionBasedCF":
        sample_size = trial.suggest_int("sample_size", 900, 5000)
        prompt_session_emb_comb_strategy = trial.suggest_categorical(
            "prompt_session_emb_comb_strategy", ["mean", "last", "concat"]
        )
        if prompt_session_emb_comb_strategy == "concat":
            training_session_emb_comb_strategy = "concat"
        else:
            training_session_emb_comb_strategy = trial.suggest_categorical(
                "training_session_emb_comb_strategy", ["mean", "last"]
            )

        sknn_emb_config = {
            "sample_size": sample_size,
            "k": trial.suggest_int("k", 10, sample_size - 1),
            "sampling": trial.suggest_categorical(
                "sampling", ["random", "recent", "idf", "idf_greedy"]
            ),
            "similarity_measure": trial.suggest_categorical(
                "similarity_measure", ["dot", "cosine"]),
            "decay": trial.suggest_categorical(
                "decay", ["linear", "log", "harmonic", "quadratic"]),
            "idf_weighting": trial.suggest_categorical("idf_weighting",
                                                       [True, False]),
            "last_n_items": trial.suggest_int("last_n_items", 10, 100, step=10),
            "training_session_decay": trial.suggest_categorical(
                "training_session_decay",
                ["linear", "log", "harmonic", "quadratic", None]
            ),
            "use_item_embeddings": True,
            "dim_reduction_config": {
                "normalize": True,
                "reduced_dim_size": trial.suggest_int("reduced_dim_size",
                                                      512, 3092, step=256),
                "reduction_config": {
                    # "reduction_technique": trial.suggest_categorical("reduction_technique", ["lda", "pca"]),
                    "reduction_technique": "pca",
                    "config": {},
                },
            },
            "prompt_session_emb_comb_strategy": prompt_session_emb_comb_strategy,
            "training_session_emb_comb_strategy": training_session_emb_comb_strategy,
            "cores": CORES,
            "filter_prompt_items": FILTER_PROMPT_ITEMS,
            "is_verbose": IS_VERBOSE,
            "max_session_length_for_decay_precomputation": MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION,
        }

        model = model_class(**sknn_emb_config)

    elif model_class.__name__ == "LLMSeqSim":
        llmseqsim_config = {
            "batch_size":trial.suggest_int("fit_batch_size", 32, 1024, step=64),
            "combination_decay": trial.suggest_categorical(
                "combination_decay", ["constant_linear", "scaling_linear",
                                      "scaling_quadratic", "scaling_cubic",
                                      "log", "harmonic", "harmonic_squared"]),
            "embedding_combination_strategy": trial.suggest_categorical(
                "embedding_combination_strategy", ["mean", "last"]),
            "similarity_measure": trial.suggest_categorical(
                "similarity_measure", ["cosine", "dot", "euclidean"]),
            "dim_reduction_config": {
                "normalize": True,
                "reduced_dim_size": trial.suggest_int("reduced_dim_size",
                                                      512, 3092, step=256),
                "reduction_config": {
                    # "reduction_technique": trial.suggest_categorical(
                    #     "reduction_technique", ["pca"]),
                    "reduction_technique": "pca",
                    "config": {},
                },
            },
            "cores": CORES,
            "filter_prompt_items": FILTER_PROMPT_ITEMS,
            "is_verbose": IS_VERBOSE,
            "max_session_length_for_decay_precomputation": MAX_SESSION_LENGTH_FOR_DECAY_PRECOMPUTATION,
        }
        model = model_class(**llmseqsim_config)

    if model_class.__name__ in ["SessionBasedCF", "LLMSeqSim"]:
        model.train(dataset_train.get_train_data(), dataset_train.get_item_data())
    else:
        model.train(dataset_train.get_train_data())

    predictions = model.predict(dataset_train.get_test_prompts(), top_k=max(TOP_Ks))
    report = Evaluation.eval(
        predictions=predictions,
        ground_truths=dataset_train.get_test_ground_truths(),
        model_name=model.name(),
        top_k=max(TOP_Ks),
        metrics=[NormalizedDiscountedCumulativeGain()],
        dependencies={
            metrics.MetricDependency.NUM_ITEMS: dataset_train.get_unique_item_count(),
        },
        metrics_per_sample=False,
    )
    trial_df = report.to_df()
    ndcg = trial_df.at[model.name(), "NDCG@20"]

    return ndcg

for model_class in model_classes:
    for embedding_key in EMBEDDING_PATHS:
        OPENAI_EMBEDDINGS_PATH = EMBEDDING_PATHS[embedding_key]
        OPENAI_DATASET_FILENAME = DATASET_PATHS[embedding_key]

        EXPERIMENTS_FOLDER = os.path.join("../results", model_class.__name__)
        os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)

        if model_class.__name__ in ["SessionBasedCF", "LLMSeqSim"]:
            dataset_train = SessionDataset.from_pickle(OPENAI_DATASET_FILENAME)
        else:
            dataset_train = SessionDataset.from_pickle(DATASET_FILENAME)

        OPTUNA_STUDY_FILE = os.path.join(EXPERIMENTS_FOLDER, f"optuna_study_{model_class.__name__}.pkl")

        if os.path.exists(OPTUNA_STUDY_FILE):
            os.remove(OPTUNA_STUDY_FILE)

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED))

        def save_study_callback(study, trial):
            if trial.number % 5 == 0:
                with open(OPTUNA_STUDY_FILE, "wb") as f:
                    pickle.dump(study, f)
                print(f"[{model_class.__name__}-{embedding_key}] Trial {trial.number} saved.")

        study.optimize(objective, n_trials=50, callbacks=[save_study_callback])

        with open(OPTUNA_STUDY_FILE, "wb") as f:
            pickle.dump(study, f)

        sorted_trials = sorted(study.trials, key=lambda x: x.value, reverse=True)
        best_trials_data = [
            {"Rank": i + 1, "Value": t.value, "Params": str(t.params)}
            for i, t in enumerate(sorted_trials[:3])
        ]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_file_path = os.path.join(
            EXPERIMENTS_FOLDER,
            f"{model_class.__name__}-{embedding_key}_best_3_{timestamp}.csv"
        )
        pd.DataFrame(best_trials_data).to_csv(csv_file_path, index=False)
        print(f"[{model_class.__name__}-{embedding_key}] Best 3 results saved to {csv_file_path}")

        if os.path.exists(OPTUNA_STUDY_FILE):
            os.remove(OPTUNA_STUDY_FILE)
            print("Optuna study file has been deleted.")