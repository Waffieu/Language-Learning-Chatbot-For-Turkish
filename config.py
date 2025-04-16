import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GPU settings
GPU_ENABLED = os.getenv("GPU_ENABLED", "true").lower() == "true"
GPU_MEMORY_FRACTION = float(os.getenv("GPU_MEMORY_FRACTION", "0.8"))
GPU_CLEAR_CACHE_INTERVAL = int(os.getenv("GPU_CLEAR_CACHE_INTERVAL", "300"))  # seconds

# Bot configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Memory settings
SHORT_MEMORY_SIZE = int(os.getenv("SHORT_MEMORY_SIZE", "25"))
LONG_MEMORY_SIZE = int(os.getenv("LONG_MEMORY_SIZE", "100"))
MEMORY_DIR = os.getenv("MEMORY_DIR", "user_memories")
MEMORY_AUTOSAVE_INTERVAL = int(os.getenv("MEMORY_AUTOSAVE_INTERVAL", "60"))  # seconds
MEMORY_CACHE_SIZE = int(os.getenv("MEMORY_CACHE_SIZE", "32"))  # number of chats to cache

# Web search settings
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "100"))

# Proxy settings - DISABLED
# Proxy system has been removed due to connection issues with DuckDuckGo
PROXY_ENABLED = False
PROXY_LIST = []
PROXY_FILE = ""

# Maximum number of retries for DuckDuckGo searches
MAX_SEARCH_RETRIES = int(os.getenv("MAX_SEARCH_RETRIES", "10"))

# Time awareness settings
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Europe/Istanbul")
TIME_AWARENESS_ENABLED = os.getenv("TIME_AWARENESS_ENABLED", "true").lower() == "true"
# Only show time information when relevant to the conversation
SHOW_TIME_ONLY_WHEN_RELEVANT = os.getenv("SHOW_TIME_ONLY_WHEN_RELEVANT", "true").lower() == "true"

# Website link settings
# Only show website links when explicitly requested or relevant
SHOW_LINKS_ONLY_WHEN_RELEVANT = os.getenv("SHOW_LINKS_ONLY_WHEN_RELEVANT", "true").lower() == "true"

# Gemini model settings
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
GEMINI_TEMPERATURE = 0.7
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 40
GEMINI_MAX_OUTPUT_TOKENS = 2048

# Specific model settings for word translation
WORD_TRANSLATION_MODEL = "gemini-2.0-flash-lite"
WORD_TRANSLATION_TEMPERATURE = 0.1
WORD_TRANSLATION_TOP_P = 0.95
WORD_TRANSLATION_TOP_K = 40
WORD_TRANSLATION_MAX_OUTPUT_TOKENS = 200

# Specific model settings for search query generation
SEARCH_QUERY_MODEL = "gemini-2.0-flash-lite"
SEARCH_QUERY_TEMPERATURE = 0.2
SEARCH_QUERY_TOP_P = 0.95
SEARCH_QUERY_TOP_K = 40
SEARCH_QUERY_MAX_OUTPUT_TOKENS = 256

# Specific model settings for language detection
LANGUAGE_DETECTION_MODEL = "gemini-2.0-flash-lite"
LANGUAGE_DETECTION_TEMPERATURE = 0.1
LANGUAGE_DETECTION_TOP_P = 0.95
LANGUAGE_DETECTION_TOP_K = 40
LANGUAGE_DETECTION_MAX_OUTPUT_TOKENS = 10

# Safety settings - all set to BLOCK_NONE as requested
SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]
