from __future__ import annotations

import logging
from functools import cached_property
from abc import ABC, abstractmethod

import pandas as pd
import polars as pl
from typing import Any
from pathlib import Path
from sklearn import metrics as sklearn_metrics
from statsforecast import StatsForecast, models as sf_models
from neuralforecast import NeuralForecast, models as nf_models

from src.config import PipelineConfig
from datetime import datetime


class Pipeline(ABC):
    _forecaster_class: Any = None
    _models_lib: Any = None

    def __init__(
        self,
        config: PipelineConfig,
    ):
        if isinstance(config.data_path, str):
            config.data_path = Path(config.data_path)

        if isinstance(config.checkpoint_path, str):
            config.checkpoint_path = Path(config.checkpoint_path)

        self.data_path = Path(config.data_path)
        self.id_column = config.id_column
        self.date_column = config.date_column
        self.target_column = config.target_column
        self.feature_columns = config.features_columns or []

        self.min_date = config.min_date
        self.max_date = config.max_date
        self.freq = config.freq

        self.h = config.h
        self.models = config.models
        self.checkpoint_path = config.checkpoint_path

        self.log_dir = Path(config.log_dir)
        experiment_name = f'{self._forecaster_class.__name__.lower()}_experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.experiment_dir / 'pipeline.log'
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s | %(levelname)s | %(name)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ],
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        config.to_yaml(self.experiment_dir / 'config.yaml')

        self.mode = config.mode
        self.metric_names = config.metric_names or ['mean_absolute_percentage_error']

        if isinstance(self.metric_names, str):
            self.metric_names = [self.metric_names]

        self._fitted = False

    @property
    def _id_column_alias(self) -> str:
        return 'unique_id'

    @property
    def _date_column_alias(self) -> str:
        return 'ds'

    @property
    def _target_column_alias(self) -> str:
        return 'y'

    @cached_property
    def _rename_dict(self) -> dict[str, str]:
        return {
            self.id_column: self._id_column_alias,
            self.date_column: self._date_column_alias,
            self.target_column: self._target_column_alias,
        }

    @abstractmethod
    @cached_property
    def _fit_kwargs(self) -> dict[str, Any]: ...

    @abstractmethod
    @cached_property
    def _predict_kwargs(self) -> dict[str, Any]: ...

    def read_data(
        self,
        filters: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        self.logger.info('Reading data from %s', self.data_path)
        if self.data_path.suffix == '.csv':
            data = pl.read_csv(self.data_path)
        elif self.data_path.suffix == '.parquet':
            read_columns = [self.id_column, self.date_column, self.target_column, *self.feature_columns]
            data = pl.read_parquet(
                self.data_path, use_pyarrow=True, columns=read_columns, pyarrow_options={'filter': filters}
            )
        else:
            raise NotImplementedError('Only CSV and parquet supported')

        self.logger.info('Data shape after reading: %s', data.shape)
        return data


    def process_data(self, data):
        self.logger.info('Processing data')
        data = data.with_columns(pl.col(self.date_column).cast(pl.Date))
        data = data.rename(self._rename_dict)
        data = data.select(
            self._id_column_alias,
            self._date_column_alias,
            self._target_column_alias,
            *self.feature_columns
        )
        self.logger.info('Data shape after processing: %s', data.shape)
        return data

    def train_test_split(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        self.logger.info('Splitting data into train and test sets')
        min_date = self.min_date or df[self._date_column_alias].min()
        max_date = self.max_date or df[self._date_column_alias].max()

        if self.mode == 'inference':
            train_df = df.filter(
                (pl.col(self._date_column_alias) > pd.to_datetime(min_date)) &
                (pl.col(self._date_column_alias) <= pd.to_datetime(max_date))
            )
            self.logger.info(
                'Train shape: %s, Test shape: %s',
                train_df.shape,
                None,
            )
            return train_df, None

        split_date = pl.lit(max_date).dt.offset_by(f'-{self.h}{self.freq}')

        train_df = df.filter(
            (pl.col(self._date_column_alias) > pd.to_datetime(min_date)) &
            (pl.col(self._date_column_alias) <= split_date)
        )
        test_df = df.filter(
            (pl.col(self._date_column_alias) > split_date) &
            (pl.col(self._date_column_alias) <= pd.to_datetime(max_date))
        )
        self.logger.info(
            'Train shape: %s, Test shape: %s',
            train_df.shape,
            None if test_df is None else test_df.shape,
        )

        return train_df, test_df

    def fit(self, data: pl.DataFrame):
        self.logger.info('Fitting models: %s', [model['name'] for model in self.models])
        if self._fitted:
            raise RuntimeError('Pipeline is already fitted')

        models = [
            getattr(self._models_lib, model['name'])(**model.get('params', {}))
            for model in self.models
        ]
        forecaster = self._forecaster_class(models=models, freq=self.freq)
        forecaster.fit(data, **self._fit_kwargs)
        self._fitted = True
        return forecaster

    def predict(self, forecaster: StatsForecast) -> pl.DataFrame:
        self.logger.info('Predicting with horizon %s', self.h)
        if not self._fitted:
            raise RuntimeError(f'Pipeline is not fitted')
        predict = forecaster.predict(**self._predict_kwargs)
        return predict

    @staticmethod
    def save(forecaster, path: str | Path) -> None:
        forecaster.save(str(path))

    @staticmethod
    @abstractmethod
    def load(path: Path): ...

    def calc_metrics(
        self,
        y_true: pl.DataFrame,
        y_pred: pl.DataFrame,
    ) -> pl.DataFrame:
        self.logger.info('Calculating metrics')
        joined = y_true.join(
            y_pred,
            on=[self._id_column_alias, self._date_column_alias],
        )

        metrics_list: list[dict[str, Any]] = []
        pred_columns: list[str] = list(set(y_pred.columns) - {self._id_column_alias, self._date_column_alias})

        for model_name in pred_columns:
            y = joined[self._target_column_alias].to_numpy()
            y_hat = joined[model_name].to_numpy()
            model_metrics = {'model': model_name}
            for metric_name in self.metric_names:
                metric_fn = getattr(sklearn_metrics, metric_name)
                model_metrics[metric_name] = metric_fn(y, y_hat)

            metrics_list.append(model_metrics)

        return pl.DataFrame(metrics_list)

    def save_artifacts(
        self,
        predict: pl.DataFrame,
        forecaster: StatsForecast,
        metrics: pl.DataFrame | None = None,
    ) -> None:
        self.logger.info('Saving artifacts to %s', self.experiment_dir)
        predict.write_parquet(self.experiment_dir / 'prediction.parquet')

        if metrics is not None:
            metrics.write_csv(self.experiment_dir / 'metrics.csv')

        if not self.checkpoint_path:
            self.save(forecaster, self.experiment_dir / 'forecaster.pk')

    def run(self):
        self.logger.info('Starting pipeline run in "%s" mode', self.mode)
        data = self.read_data()
        data = self.process_data(data)
        train_df, test_df = self.train_test_split(data)
        if self.checkpoint_path:
            forecaster = self.load(self.checkpoint_path)
        else:
            forecaster = self.fit(train_df)

        predict = self.predict(forecaster)

        metrics = None
        if self.mode == 'valid':
            metrics = self.calc_metrics(test_df, predict)
            self.logger.info('Prediction metrics')
            self.logger.info("\n%s", metrics.to_pandas().to_markdown(index=False))

        self.save_artifacts(predict, forecaster, metrics)
        self.logger.info('Pipeline run completed')


class StatsForecastPipeline(Pipeline):
    _forecaster_class = StatsForecast
    _models_lib = sf_models

    @cached_property
    def _fit_kwargs(self) -> dict[str, Any]:
        return {}

    @cached_property
    def _predict_kwargs(self) -> dict[str, Any]:
        return {'h': self.h}

    @staticmethod
    def load(path: Path) -> StatsForecast:
        return StatsForecast.load(path)


class NeuralForecastPipeline(Pipeline):
    _forecaster_class = NeuralForecast
    _models_lib = nf_models

    @cached_property
    def _fit_kwargs(self) -> dict[str, Any]:
        return {'val_size': self.h}

    @cached_property
    def _predict_kwargs(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def load(path: Path) -> NeuralForecast:
        return NeuralForecast.load(str(path))
