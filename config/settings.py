from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    
    # Data paths relative to base_dir
    data_path: str = "data/qcom_pune_dataset.csv"
    model_path: str = "models/conversion_model.pkl"
    
    # Pricing configuration
    fee_min: int = 0
    fee_max: int = 100
    fee_step: int = 5
    conversion_drop_budget: float = 0.03

    @property
    def get_data_path(self) -> Path:
        return self.base_dir / self.data_path

    @property
    def get_model_path(self) -> Path:
        return self.base_dir / self.model_path

    class Config:
        env_file = ".env"

settings = Settings()
