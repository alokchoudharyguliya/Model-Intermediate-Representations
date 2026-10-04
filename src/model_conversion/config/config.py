import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR=Path(__file__).resolve().parent
_env_artifact_path=os.getenv("ARTIFACT_PATH","./artifacts")
ARTIFACT_PATH=Path(_env_artifact_path)
if not ARTIFACT_PATH.is_absolute():
    ARTIFACT_PATH=(ROOT_DIR/ARTIFACT_PATH).resolve()

ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)

