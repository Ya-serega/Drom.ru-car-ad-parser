import re
import emoji
import numpy as np
import pandas as pd
import pymorphy2
from typing import List

"""Rule-based извлечение интерпретируемых признаков
Просто поиск ключевых слов, логических выражений"""

dealer_keywords = ["кредит", "автосалон", "трейд-ин", "гарантия", "поможем оформить", "рассрочка"]
reseller_keywords = ["без вложений", "сел и поехал", "срочно", "вложений не требует", "торг у капота", "не бит не крашен"]
tuning_keywords = ["stage", "st-", "стэйдж", "турбина", "чип", "выхлоп", "прошивка", "тюнинг", "злой",
                   "ковши", "усиленный"]
accident_keywords = ["бит", "крашен", "после ДТП", "окрас", "замена крыла",
                              "притертость", "переварка", "переварены"]
urgent_keywords = ["срочно", "торг", "обмен", "скину", "цена до"]
risky_keywords = reseller_keywords + tuning_keywords + ["бит", "после ДТП", "было ДТП", "после аварии"]

morph = pymorphy2.MorphAnalyzer()


def safe_text(text):
    if pd.isna(text):
        return ""
    return str(text)


def contains_keywords(text, keywords):
    text = safe_text(text).lower()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return 1
    return 0


def count_caps_words(text: str, min_length: int = 2) -> int:
    """
    Подсчет слов в верхнем регистре с минимальной длиной.

    Args:
        text: входной текст
        min_length: минимальная длина слова для учета

    Returns:
        int: количество слов в верхнем регистре длиной >= min_length
    """
    if not text:
        return 0

    words = re.findall(r'\b\w+\b', text)

    return sum(1 for word in words if word.isupper() and len(word) >= min_length)


def count_phrases_with_morphs(text: str, phrases: List[str]) -> int:
    """
    Проверяем вхождения слов с учетом морфологии, приводим к нормальной форме (лемматизация).
    Считает все вхождения, включая пересекающиеся.
    """

    # Лемматизация текста
    text = safe_text(text)
    text_words = re.findall(r'\b\w+\b', text)
    text_lemmas = [morph.parse(word)[0].normal_form for word in text_words]
    text_str = ' '.join(text_lemmas)

    # Лемматизация искомых фраз
    target_phrases = []
    total = 0
    for phrase in phrases:
        phrase_words = phrase.lower().split()
        phrase_lemmas = [morph.parse(word)[0].normal_form for word in phrase_words]
        target = ' '.join(phrase_lemmas)
        total += text_str.count(target)

    return total


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_series = df["raw_text"].fillna("").astype(str)
    features = pd.DataFrame(index=df.index)

    features["is_dealer_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, dealer_keywords)
    )

    features["is_reseller_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, reseller_keywords)
    )

    features["is_tuned_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, tuning_keywords)
    )

    features["is_urgent_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, urgent_keywords)
    )

    features["was_beaten_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, accident_keywords)
    )

    features["num_risk_keywords"] = text_series.apply(
        lambda x: count_phrases_with_morphs(x, risky_keywords)
    )

    features["num_caps_words"] = text_series.apply(count_caps_words)

    features["num_exclamation_marks"] = text_series.str.count("!")

    features["num_emoji"] = text_series.apply(emoji.emoji_count)

    features["text_length"] = text_series.str.len()

    return features


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    # Логарифм цены
    features["log_price"] = np.log1p(df["price"])

    # Возраст авто
    parse_year = pd.to_datetime(df["parse_date"]).dt.year
    car_age = parse_year - df["year"]
    features["car_age"] = car_age

    # Цена за л.с.
    engine_hp = df["engine_hp"].replace(0, np.nan)
    features["price_per_hp"] = df["price"] / engine_hp

    # Пробег в год
    features["mileage_per_year"] = np.where(
        car_age > 0,
        df["mileage_km"] / car_age,
        df["mileage_km"]
    )

    return features


def build_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    features["seller_is_dealer"] = (df["seller_type"] == "dealer").astype(int)
    features["seller_is_owner"] = (df["seller_type"] == "owner").astype(int)

    features["is_good_deal"] = df["drom_price_cat"].isin(
        ["Хорошая цена", "Отличная цена"]
    ).astype(int)

    return features


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    text_f = build_text_features(df)
    num_f = build_numeric_features(df)
    cat_f = build_categorical_features(df)

    all_features = pd.concat([text_f, num_f, cat_f], axis=1)

    return all_features
