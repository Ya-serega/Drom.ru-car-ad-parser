import re
from bs4 import BeautifulSoup


# Базовая очистка
def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text(" ")


# Нормализация пробелов
def normalize_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Удаление телефонов
def remove_phones(text):
    phone_pattern = r'(\+7|8)?[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
    return re.sub(phone_pattern, ' PHONE ', text)


# Удаление ссылок и спам-символов
def remove_links(text):
    return re.sub(r'http\S+|www\S+', ' LINK ', text)


def remove_repeated_symbols(text):
    return re.sub(r'(.)\1{3,}', r'\1\1', text)


def clean_text(raw_text):
    if not raw_text:
        return ""
    text = raw_text.lower()
    text = clean_html(text)
    text = remove_links(text)
    text = remove_phones(text)
    text = remove_repeated_symbols(text)
    text = normalize_spaces(text)
    return text
