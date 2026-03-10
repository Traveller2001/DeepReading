"""Centralised LLM configuration.  Edit this file to adapt to your own LLM provider."""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ---- Project paths ----
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# ---- Main LLM (report generation, discussion, etc.) ----
LLM_API_BASE      = os.environ.get("LLM_API_BASE", "")
LLM_API_KEY       = os.environ.get("LLM_API_KEY", "")
LLM_API_VERSION   = os.environ.get("LLM_API_VERSION", "")   # 设置此项则使用 Azure 模式
LLM_MODEL         = os.environ.get("LLM_MODEL", "deepseek-chat")
REPORT_MODEL      = os.environ.get("REPORT_MODEL", LLM_MODEL)
LLM_TEMPERATURE   = 0.7
LLM_TIMEOUT       = 180        # 秒
MAX_REPORT_TOKENS = 65536
MAX_TOOL_ROUNDS   = 25
MAX_TOOL_RESULT_LEN = 8000
MAX_CONTINUATIONS = 3

# ---- Vision LLM (figure extraction from PDF pages) ----
VISION_MODEL       = os.environ.get("VISION_MODEL", LLM_MODEL)
VISION_TEMPERATURE = 0.1
VISION_MAX_TOKENS  = 4096
VISION_TIMEOUT     = 120
VISION_BATCH_SIZE  = 4
VISION_RETRY_DELAY = 3

# ---- PDF rendering ----
SCAN_DPI  = 150
CROP_DPI  = 200
BBOX_PAD  = 5
