import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import polars as pl

from config import PipelineConfig
from pipeline import NeuralForecastPipeline, StatsForecastPipeline


pl.Config(
    float_precision=6,
    thousands_separator='_'
)

config_path = '../conf/statsforecast.yaml'
config = PipelineConfig.from_yaml(config_path)
pipeline = StatsForecastPipeline(config)
pipeline.run()
