import logging

from dotenv import load_dotenv

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot_logs.txt"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# Загрузка переменных окружения
load_dotenv()

# API ключ

# Пути
RESULTS_DIR = "../results"


# Параметры обработки
WHISPER_MODEL_NAME = "large-v2"
DEVICE = "cpu"
BATCH_SIZE = 4
SUPPRESS_NUMERALS = False # True/False = заменять ли цифры словами
LANGUAGE = None  # None = автоопределение

