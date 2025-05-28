from typing import Any, Literal
from pathlib import Path

import yaml

from pydantic.v1 import BaseSettings


class PipelineConfig(BaseSettings):
    h: int
    models: str | list[str]
    models_params: list[dict[str, Any]]
    data_path: Path | str
    id_column: str
    date_column: str
    target_column: str
    freq: str
    mode: Literal['valid', 'inference'] = 'valid'
    features_columns: list[str] | None = None
    checkpoint_path: Path | str | None = None
    min_date: str | None = None
    max_date: str | None = None
    metric_names: str | list[str] | None = None

    class Config:
        # env_prefix = 'PIPELINE_'
        case_sensitive = False

    @classmethod
    def from_yaml(cls, config_path: str | Path):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f'Config file `{config_path}` not found.')
        with config_file.open('r') as f:
            data = yaml.safe_load(f)

        return cls(**data)
