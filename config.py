
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class SystemConfig:
    api_key: str
    model_name: str = "gemini-1.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 30
    log_level: str = "INFO"
    
    # Evaluation thresholds
    min_accuracy_score: float = 0.7
    min_completeness_score: float = 0.8
    max_response_time: float = 5.0
