from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GLASSES_", env_file=".env")

    data_dir: Path = Path("data")
    max_upload_bytes: int = 15 * 1024 * 1024
    max_image_pixels: int = 24_000_000
    host: str = "127.0.0.1"
    port: int = 8000
    fake_training: bool = False


def load_settings(data_dir: Path | None = None) -> Settings:
    values = {"data_dir": data_dir} if data_dir is not None else {}
    return Settings(**values)
