from typing import Any, Literal, Type
from pathlib import Path

import yaml

from pydantic.v1 import BaseSettings, validator


class PipelineConfig(BaseSettings):
    h: int
    models: list[dict[str, Any]]
    data_path: Path | str
    id_column: str
    date_column: str
    target_column: str
    freq: str
    mode: Literal['valid', 'inference'] = 'valid'
    log_dir: Path | str
    features_columns: list[str] | None = None
    checkpoint_path: Path | str | None = None
    min_date: str | None = None
    max_date: str | None = None
    metric_names: str | list[str] | None = None
    boxcox: bool | None = False

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

    def to_yaml(self, save_path: str | Path):
        save_path = Path(save_path)

        str_config = {k: str(v) for k,v in self.dict().items()}

        with save_path.open('w') as yaml_file:
            yaml.safe_dump(str_config, yaml_file, sort_keys=False)
