from pipeline import Pipeline
from config import PipelineConfig, get_pipeline_config

config: PipelineConfig = get_pipeline_config()
pipeline = Pipeline.from_config(config)
pipeline.run()
