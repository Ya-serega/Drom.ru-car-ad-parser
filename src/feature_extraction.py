"""Модуль извлечения признаков на основе правил для данных объявлений.

Генерирует интерпретируемые признаки из текстовых, числовых и категориальных
колонок. Предназначен для задач оценки стоимости автомобилей, профилирования
продавцов и детекции мошеннических объявлений.
"""
import re
import emoji
import numpy as np
import pandas as pd
import pymorphy2
from typing import List
from src.config import (DEALER_KEYWORDS, RESELLER_KEYWORDS, TUNING_KEYWORDS, ACCIDENT_KEYWORDS,
                        URGENT_KEYWORDS, RISKY_KEYWORDS)

morph = pymorphy2.MorphAnalyzer()


def safe_text(text):
    """
    Безопасное преобразование входного значения в строку с обработкой пропусков.

    Args:
        text: Входное значение (строка, число, NaN или None).

    Returns:
        str: Строковое представление. Возвращает пустую строку для null-значений.
    """
    if pd.isna(text):
        return ""
    return str(text)


def contains_keywords(text, keywords):
    """
    Проверяет наличие хотя бы одного ключевого слова из списка.

    Использует регулярные выражения с границами слов (\b) для исключения
    частичных совпадений.

    Args:
        text: Входной текст для поиска.
        keywords: Итерируемый объект с целевыми ключевыми словами.

    Returns:
        int: 1, если найдено совпадение, иначе 0.
    """
    text = safe_text(text).lower()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return 1
    return 0


def count_caps_words(text: str, min_length: int = 2) -> int:
    """
    Подсчитывает слова, написанные полностью заглавными буквами.
    Полезно для детекции эмоционального форматирования или спам-объявлений от автосалонов.

    Args:
        text: Входной текст.
        min_length: Минимальная длина слова для учета.

    Returns:
        int: Количество слов в верхнем регистре длиной >= min_length.
    """
    if not text:
        return 0

    words = re.findall(r'\b\w+\b', text)

    return sum(1 for word in words if word.isupper() and len(word) >= min_length)


def count_phrases_with_morphs(text: str, phrases: List[str]) -> int:
    """
    Подсчитывает вхождения целевых фраз с учетом морфологии.
    Приводит текст и искомые фразы к нормальной форме (лемматизация)
    через `pymorphy2`, что позволяет обрабатывать разные падежи, числа и роды.

    Args:
        text: Входной текст для анализа.
        phrases: Список целевых фраз.

    Returns:
        int: Общее количество найденных лемматизированных фраз.
    """

    # Лемматизация текста
    text = safe_text(text)
    text_words = re.findall(r'\b\w+\b', text)
    text_lemmas = [morph.parse(word)[0].normal_form for word in text_words]
    text_str = ' '.join(text_lemmas)

    # Лемматизация искомых фраз
    total = 0
    for phrase in phrases:
        phrase_words = phrase.lower().split()
        phrase_lemmas = [morph.parse(word)[0].normal_form for word in phrase_words]
        target = ' '.join(phrase_lemmas)
        total += text_str.count(target)

    return total


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Извлекает текстовые признаки на основе заданных списков ключевых слов.

    Генерирует бинарные флаги (дилер, перекуп, тюнинг, ДТП, срочно),
    счетчики рискованных фраз, метрики регистра, пунктуации, эмодзи и длины текста.

    Args:
        df: DataFrame, содержащий колонку 'raw_text'.

    Returns:
        pd.DataFrame: DataFrame с извлеченными текстовыми признаками,
        индексированный аналогично входному.
    """
    df = df.copy()
    text_series = df["raw_text"].fillna("").astype(str)
    features = pd.DataFrame(index=df.index)

    features["is_dealer_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, DEALER_KEYWORDS)
    )

    features["is_reseller_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, RESELLER_KEYWORDS)
    )

    features["is_tuned_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, TUNING_KEYWORDS)
    )

    features["is_urgent_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, URGENT_KEYWORDS)
    )

    features["was_beaten_by_rule"] = text_series.apply(
        lambda x: contains_keywords(x, ACCIDENT_KEYWORDS)
    )

    features["num_risk_keywords"] = text_series.apply(
        lambda x: count_phrases_with_morphs(x, RISKY_KEYWORDS)
    )

    features["num_caps_words"] = text_series.apply(count_caps_words)

    features["num_exclamation_marks"] = text_series.str.count("!")

    features["num_emoji"] = text_series.apply(emoji.emoji_count)

    features["text_length"] = text_series.str.len()

    return features


def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет производные числовые признаки из сырых колонок.

    Рассчитывает логарифм цены, возраст автомобиля, цену за лошадиную силу
    и среднегодовой пробег. Корректно обрабатывает деление на ноль и пропуски.

    Args:
        df: DataFrame с колонками 'price', 'parse_date', 'year', 'engine_hp', 'mileage_km'.

    Returns:
        pd.DataFrame: DataFrame с вычисленными числовыми признаками.
    """
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
    """
    Создает бинарные признаки из категориальных колонок.

    Кодирует тип продавца и категорию цены на Drom в машинно-читаемые флаги.

    Args:
        df: DataFrame с колонками 'seller_type' и 'drom_price_cat'.

    Returns:
        pd.DataFrame: DataFrame с бинарными категориальными признаками.
        """
    df = df.copy()
    features = pd.DataFrame(index=df.index)

    features["seller_is_dealer"] = (df["seller_type"] == "dealer").astype(int)
    features["seller_is_owner"] = (df["seller_type"] == "owner").astype(int)

    features["is_good_deal"] = df["drom_price_cat"].isin(
        ["Хорошая цена", "Отличная цена"]
    ).astype(int)

    return features


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Оркестрирует извлечение признаков по всем типам данных.

    Объединяет текстовые, числовые и категориальные признаки в единую
    матрицу, выровненную по индексу входного DataFrame.

    Args:
        df: Входной DataFrame, содержащий все необходимые исходные колонки.

    Returns:
        pd.DataFrame: Итоговая матрица признаков, готовая для обучения моделей.
    """
    text_f = build_text_features(df)
    num_f = build_numeric_features(df)
    cat_f = build_categorical_features(df)

    all_features = pd.concat([text_f, num_f, cat_f], axis=1)

    return all_features
