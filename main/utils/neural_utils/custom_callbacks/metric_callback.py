from typing import Union, Any

import numpy as np
from keras import callbacks

from main.eval.metrics.metric import (
    RankingMetric,
    MetricDependency,
)
from main.abstract_model import Model


class MetricCallback(callbacks.Callback):
    def __init__(
        self,
        main_model: Model,
        metric_cls: RankingMetric,
        predict_data: Union[np.ndarray, dict[int, np.ndarray], None],
        ground_truths: dict[int, np.ndarray],
        top_k: int,
        prefix: str = "",
        dependencies: dict[MetricDependency, Any] = {},
        cores: int = 1,
    ):
        """A callback to evaluate the model on a given metric every epoch.

        Args:
            main_model (Model): The model to evaluate.
            metric_cls (RankingMetric): The metric class to evaluate.
            predict_data (Union[np.ndarray, dict[int, np.ndarray], None]):
                The data that is passed to the predict method of the model.
            ground_truths (dict[int, np.ndarray]): The ground-truths corresponding
                to the predict_data, which is used to calculate the metric.
            top_k (int): The size of the recommendation slate.
            prefix (str): Prefix to the name of the metric for logging. Defaults to "".
            dependencies (dict[MetricDependency, Any], optional): The metric
                dependencies that are necessary for the metric to compute its values.
                Defaults to {}.
            cores (int, optional): The number of cores for evaluation. Defaults to 1.
        """
        super().__init__()

        self.main_model = main_model
        self.metric_cls = metric_cls
        self.predict_data = predict_data
        self.ground_truths = ground_truths
        self.top_k = top_k
        self.prefix = prefix
        self.dependencies = dependencies
        self.cores = cores

        main_model.is_trained = True

    def on_epoch_end(self, epoch, logs=None):
        # Predict.
        predictions = self.main_model.predict(self.predict_data, self.top_k)

        result = self.metric_cls.eval(
            predictions=predictions,
            ground_truths=self.ground_truths,
            top_k=self.top_k,
            dependencies=self.dependencies,
            cores=self.cores,
        )

        # Prepare log.
        # Since .name requires member variables, we instantiate a temporary metric
        # here to get the name, and use it as the key in our logs.
        temp_metric = self.metric_cls()
        temp_metric.top_k = self.top_k
        logs[f"{self.prefix}{temp_metric.name()}"] = result

        metric_name = f"{self.prefix}{temp_metric.name()}"
        if logs is not None:
            logs[metric_name] = result  # Add result to logs

        print(f"check NDCG@{self.top_k}")
        # Update the best NDCG@20 score in the main model.
        if result > self.main_model.best_ndcg:
            self.main_model.best_ndcg = result  # Update the best score
            print('updated!')
