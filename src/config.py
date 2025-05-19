from functools import lru_cache
from typing import Optional, Literal
from pathlib import Path

from pydantic.v1 import BaseSettings


class PipelineConfig(BaseSettings):
    h: int
    data_path: Path
    id_column: str
    date_column: str
    target_column: str
    min_date: str | None = None
    max_date: str | None = None
    freq: Literal['d', 'w', 'mo', 'q', 'y'] | None = None

    class Config:
        # env_prefix = 'PIPELINE_'
        case_sensitive = False

@lru_cache
def get_pipeline_config(**kwargs) -> PipelineConfig:
    return PipelineConfig(**kwargs)
