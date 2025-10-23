"""Runtime configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MRC_", env_file=".env", extra="ignore")

    n_regimes: int = 4
    hmm_components: int = 4
    walk_forward_train_days: int = 504
    walk_forward_test_days: int = 63
    random_state: int = 42
