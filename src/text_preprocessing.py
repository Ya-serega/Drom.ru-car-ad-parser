"""
Утилиты предварительной обработки и нормализации текста объявлений.

Предоставляет детерминированный пайплайн для очистки HTML/текста, удаления
контактных данных и ссылок, нормализации форматирования и подготовки текста
к извлечению признаков или NLP-задачам.
"""
import re
from bs4 import BeautifulSoup


# Базовая очистка
def clean_html(text):
    """
    Извлекает чистый текст из HTML-разметки.

    Args:
        text: Сырая строка, содержащая HTML-теги.

    Returns:
        str: Текст с удаленными тегами и сохраненными структурными пробелами.
    """
    return BeautifulSoup(text, "html.parser").get_text(" ")


# Нормализация пробелов
def normalize_spaces(text):
    """
    Извлекает чистый текст из HTML-разметки.

    Args:
        text: Сырая строка, содержащая HTML-теги.

    Returns:
        str: Текст с удаленными тегами и сохраненными структурными пробелами.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Удаление телефонов
def remove_phones(text):
    """
    Обнаруживает и заменяет шаблоны телефонных номеров на маркер.

    Поддерживает распространенные форматы телефонов РФ и международные коды.

    Args:
        text: Входной текст.

    Returns:
        str: Текст, где номера заменены на ' PHONE '.
    """
    phone_pattern = r'(\+7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
    return re.sub(phone_pattern, ' PHONE ', text)


# Удаление ссылок и спам-символов
def remove_links(text):
    """
    Обнаруживает и заменяет URL на плейсхолдер.

    Args:
        text: Входной текст.

    Returns:
        str: Текст, где HTTP/HTTPS/WWW ссылки заменены на ' LINK '.
    """
    return re.sub(r'http\S+|www\S+', ' LINK ', text)


def remove_repeated_symbols(text):
    """
    Сокращает избыточные повторения символов до двух подряд.

    Нормализует эмоциональную пунктуацию (например, '!!!') и спам-форматирование.

    Args:
        text: Входной текст.

    Returns:
        str: Текст со сжатыми повторяющимися символами.
    """
    return re.sub(r'(.)\1{3,}', r'\1\1', text)


def clean_text(raw_text):
    """
    Запускает полный пайплайн очистки текста.

    Применяет перевод в нижний регистр, удаление HTML, ссылок, телефонов,
    нормализацию повторов символов и пробелов в строго заданном порядке.

    Args:
        raw_text: Сырое описание объявления.

    Returns:
        str: Полностью предобработанный текст, готовый к извлечению признаков.
    """
    if not raw_text:
        return ""
    text = raw_text.lower()
    text = clean_html(text)
    text = remove_links(text)
    text = remove_phones(text)
    text = remove_repeated_symbols(text)
    text = normalize_spaces(text)
    return text
