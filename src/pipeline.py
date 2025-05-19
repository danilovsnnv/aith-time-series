from __future__ import annotations

from functools import cached_property

import polars as pl
from typing import Any, Literal
from pathlib import Path

from pydantic.v1 import BaseSettings
from sklearn import metrics as sklearn_metrics
from statsforecast import StatsForecast


class Pipeline:
    def __init__(
        self,
        h: int,
        data_path: Path | str,
        id_column: str,
        date_column: str,
        target_column: str,
        features_columns: list[str] | None = None,
        min_date: str | None = None,
        max_date: str | None = None,
        freq: Literal['d', 'w', 'mo', 'q', 'y'] | None = None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)

        self.data_path = data_path
        self.id_column = id_column
        self.date_column = date_column
        self.target_column = target_column
        self.feature_columns = features_columns or []

        self.min_date = min_date
        self.max_date = max_date
        self.freq = freq

        self.h = h

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

    @classmethod
    def from_config(cls, config: BaseSettings) -> Pipeline:
        return Pipeline(**config.dict())

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Pipeline:
        return Pipeline(**config)

    def read_data(
        self,
        filters: dict[str, Any] | None = None,
    ) -> pl.DataFrame:
        if not filters:
            filters = {}

        data = pl.read_csv(self.data_path, use_pyarrow=True, pyarrow_options={'filters': filters})
        return data


    def process_data(self, data):
        data = data.with_columns(pl.col(self.date_column).cast(pl.Date))
        data = data.rename({})

    def train_test_split(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        min_date = self.min_date or df[self.date_column].min()
        max_date = self.max_date or df[self.date_column].max()

        split_date = df.select(
            (max_date - pl.duration(**{self.freq: self.h}))
        ).item(0, 0)

        train_df = df.filter(
            (pl.col(self.date_column) > min_date) &
            (pl.col(self.date_column) <= split_date)
        )
        test_df = df.filter(
            (pl.col(self.date_column) > split_date) &
            (pl.col(self.date_column) <= max_date)
        )

        return train_df, test_df

    def fit(self, data: pl.DataFrame) -> StatsForecast:
        if self._fitted:
            raise RuntimeError(f'Pipeline is already fitted')
        forecaster = StatsForecast(models=[], freq=self.freq)
        forecaster.fit(data)
        self._fitted = True
        return forecaster

    def predict(self, forecaster: StatsForecast) -> pl.DataFrame:
        if self._fitted:
            raise RuntimeError(f'Pipeline is already fitted')
        predict = forecaster.predict(h=self.h)
        return predict

    @staticmethod
    def save(forecaster: StatsForecast, path: str | Path) -> None:
        forecaster.save(path)

    @staticmethod
    def load(path: Path) -> StatsForecast:
        return StatsForecast.load(path)

    def calc_metrics(
        self,
        y_true: pl.DataFrame,
        y_pred: pl.DataFrame,
        metric_names: str | list[str] = 'mean_absolute_percentage_error',
    ) -> pl.DataFrame:
        if isinstance(metric_names, str):
            metric_names = [metric_names]

        joined = y_true.join(
            y_pred,
            on=[self.id_column, self.date_column],
            suffix='_pred'
        )

        y = joined[self.target_column].to_numpy()
        y_hat = joined[f'{self.target_column}_pred'].to_numpy()

        metrics_dict = {}

        for metric in metric_names:
            metric_fn = getattr(sklearn_metrics, metric)
            metrics_dict[metric] = metric_fn(y, y_hat)

        metrics_df = pl.DataFrame(metrics_dict)

        return metrics_df

    def run(self):
        data = self.read_data()
        train_df, test_df = self.train_test_split(data)
        forecaster = self.fit(train_df)
        predict = self.predict(forecaster)
        metrics = self.calc_metrics(test_df, predict)
        print(metrics)

