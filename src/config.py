import os
from typing import List
from pathlib import Path

# Базовые пути
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_EXT_DIR = DATA_DIR / "external"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

TOWNS_JSONL_PATH = DATA_EXT_DIR / "towns.jsonl"
LIST_SAVE_PATH = DATA_RAW_DIR / "list_records.csv"
ADS_SAVE_PATH = DATA_RAW_DIR / "ads_records.csv"
RESULT_SAVE_PATH = DATA_PROCESSED_DIR / "dataset_final.parquet"
SESSIONS_DIR = DATA_RAW_DIR / "sessions"
HISTORY_IDS_PATH = DATA_RAW_DIR / "history_ids.txt"

# HTTP заголовки
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://auto.drom.ru/"
}

# Ссылки для парсинга (можно расширять вручную или генерировать динамически)
TARGET_URLS: List[str] = [
    "https://auto.drom.ru/opel/astra/?ph=1&pts=2&unsold=1&minprobeg=8000&whereabouts[]=0",
    "https://auto.drom.ru/toyota/corolla/?ph=1&pts=2&unsold=1&minprobeg=8000&whereabouts[]=0",
    "https://auto.drom.ru/bmw/5-series/page13/?ph=1&pts=2&unsold=1&minprobeg=8000&whereabouts[]=0"
]

# Ключевые слова для Feature Extraction
DEALER_KEYWORDS = ["кредит", "автосалон", "трейд-ин", "гарантия", "поможем оформить", "рассрочка"]
RESELLER_KEYWORDS = ["без вложений", "сел и поехал", "срочно", "вложений не требует", "торг у капота", "не бит не крашен"]
TUNING_KEYWORDS = ["stage", "st-", "стэйдж", "турбина", "чип", "выхлоп", "прошивка", "тюнинг", "злой", "ковши", "усиленный"]
ACCIDENT_KEYWORDS = ["бит", "крашен", "после ДТП", "окрас", "замена крыла", "притертость", "переварка", "переварены"]
URGENT_KEYWORDS = ["срочно", "торг", "обмен", "скину", "цена до"]
RISKY_KEYWORDS = RESELLER_KEYWORDS + TUNING_KEYWORDS + ["бит", "после ДТП", "было ДТП", "после аварии"]

# Все марки из нескольких слов (для корректного парсинга заголовка, где разделители - пробелы)
MULTI_WORD_BRANDS = ["Alfa Romeo", "Aston Martin", "Land Rover", "Rolls Royce", "Great Wall", "Mini Cooper"]


def init_dirs() -> None:
    """Создает необходимые директории при запуске"""
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(DATA_EXT_DIR, exist_ok=True)
