import polars as pl

pl.Config(
    float_precision=6,
    thousands_separator='_'
)

from pipeline import Pipeline
from config import PipelineConfig

config_path = '../conf/base.yaml'
config = PipelineConfig.from_yaml(config_path)
pipeline = Pipeline.from_config(config)
pipeline.run()
