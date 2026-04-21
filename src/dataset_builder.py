import pandas as pd
import re
import logging
import json
import os
from typing import Union
from src.text_preprocessing import clean_text
from src.feature_extraction import build_all_features
import src.config as config

logger = logging.getLogger(__name__)


def parse_price(price_text):
    if not price_text: return None
    digits = re.sub(r"\D", "", price_text)
    return int(digits) if digits else None


def parse_mileage(text):
    if not text: return None
    if "новый автомобиль" in text.lower(): return 0
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def parse_engine(text):
    if not text: return None, None
    # Исправлен regex: убраны пробелы внутри processed-string
    vol = re.search(r"(\d\.\d)", text)
    volume = float(vol.group()) if vol else None

    text_lower = text.lower()
    if "бенз" in text_lower: fuel = "petrol"
    elif "диз" in text_lower: fuel = "diesel"
    elif "гибрид" in text_lower: fuel = "hybrid"
    elif "электро" in text_lower: fuel = "electric"
    else: fuel = None
    return volume, fuel


def parse_brand_model_year(header: str):
    multi_word_brands = [b.lower() for b in config.MULTI_WORD_BRANDS]
    if not header: return None, None, None

    header = header.strip()
    pattern = r"Продажа\s+(.+?),\s*(\d{4})\s+год"
    match = re.search(pattern, header, re.IGNORECASE)

    if not match: return None, None, None

    full_name = match.group(1).strip()
    year = int(match.group(2))
    brand = model = None

    for brand_name in multi_word_brands:
        if full_name.lower().startswith(brand_name):
            brand = brand_name.title()
            model = full_name[len(brand_name):].strip() or None
            break

    if brand is None:
        parts = full_name.split()
        brand = parts[0].capitalize()
        model = " ".join(parts[1:]).strip() or None

    return brand, model, year


def parse_seller_age(text):
    if not text: return None
    years = re.search(r"\d+", text)
    return int(years.group()) if years else None


def parse_engine_hp(text):
    if not text: return None
    digits = re.sub(r"\D", "", text)
    return int(digits) if digits else None


def parse_city_name(city_name: str) -> Union[str, None]:
    if not city_name or not isinstance(city_name, str): return None
    return re.sub(r"\s+", " ", city_name.split(",")[0].strip())


def load_city_region_mapping(jsonl_file_path: str) -> dict:
    if not os.path.exists(jsonl_file_path):
        logger.warning(f"City mapping file not found: {jsonl_file_path}")
        return {}
    city_to_region = {}
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                if line.startswith('"') and line.endswith('"'): line = line[1:-1]
                data = json.loads(line)
                city = data.get("city", "").strip().lower()
                region = data.get("region_name", "").strip()
                if city and region: city_to_region[city] = region
            except json.JSONDecodeError: continue
    return city_to_region


def map_city_to_region(city_name: str, city_to_region: dict) -> str:
    if not city_name or not city_name.strip(): return "Unknown"
    city_name = city_name.strip()
    if "," in city_name:
        parts = [p.strip() for p in city_name.split(",")]
        if len(parts) >= 2 and parts[1]: return parts[1]
    region = city_to_region.get(city_name.lower())
    return region if region else f"Unknown ({city_name})"


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = ['transmission_type', 'drive_type', 'color', 'wheel_side',
                'generation', 'complectation', 'seller_type', 'drom_price_cat',
                'fuel_type', 'region', 'brand', 'model']
    for col in cat_cols:
        if col in df.columns and df[col].nunique() < df.shape[0] * 0.5:
            df[col] = df[col].astype('category')

    num_types = {
        'price': 'float32', 'log_price': 'float32', 'price_per_hp': 'float32',
        'year': 'Int16', 'car_age': 'Int16', 'num_owners_from_vin': 'Int8',
        'engine_hp': 'Int16', 'engine_vol': 'float32', 'mileage_per_year': 'float32',
        'mileage_km': 'float32', 'position_in_search': 'Int32',
        'is_dealer_by_rule': 'Int8', 'is_reseller_by_rule': 'Int8',
        'is_tuned_by_rule': 'Int8', 'is_urgent_by_rule': 'Int8',
        'was_beaten_by_rule': 'Int8', 'seller_is_dealer': 'Int8',
        'is_good_deal': 'Int8', 'num_risk_keywords': 'Int16',
        'num_caps_words': 'Int16', 'num_exclamation_marks': 'Int16',
        'num_emoji': 'Int16', 'text_length': 'Int32'
    }
    for col, dtype in num_types.items():
        if col in df.columns:
            try: df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
            except Exception as e: logger.warning(f"Type conv failed for {col}: {e}")

    for col in ['list_parse_date', 'parse_date']:
        if col in df.columns: df[col] = pd.to_datetime(df[col])
    for col in ['raw_text', 'clean_text', 'num_owners_from_specs']:
        if col in df.columns: df[col] = df[col].astype('string')
    return df


def build_dataset(df_ads: pd.DataFrame, df_list: pd.DataFrame) -> pd.DataFrame:
    # Гарантируем одинаковый тип ключа слияния
    df_ads["ad_id"] = df_ads["ad_id"].astype(str)
    df_list["ad_id"] = df_list["ad_id"].astype(str)

    city_dict = load_city_region_mapping(config.TOWNS_JSONL_PATH)
    df = pd.merge(df_list, df_ads, on="ad_id", how="inner")

    df["mileage_km"] = df["mileage_km"].apply(parse_mileage)
    df["seller_reg_time"] = df["seller_reg_time"].apply(parse_seller_age)
    df["engine_hp"] = df["engine_hp"].apply(parse_engine_hp)
    df["price"] = df["price"].apply(parse_price)

    df[["brand", "model", "year"]] = df["header"].apply(lambda x: pd.Series(parse_brand_model_year(x)))
    df[["engine_vol", "fuel_type"]] = df["engine_vol"].apply(lambda x: pd.Series(parse_engine(x)))
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df["region"] = df["city"].apply(lambda x: map_city_to_region(x, city_dict))
    df["city"] = df["city"].apply(parse_city_name)

    features = build_all_features(df)
    df = pd.concat([df, features], axis=1)

    return optimize_dtypes(df)
