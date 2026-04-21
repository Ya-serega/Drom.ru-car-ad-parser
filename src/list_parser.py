"""
Парсинг страниц выдачи поиска drom.ru.

Извлекает превью-данные объявлений (ID, URL, цена, заголовок, позиция),
а также определяет URL следующей страницы пагинации для автоматического
обхода листинга.
"""
import re
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Union, Optional, List


def safe_get_text(parent, tag: str, **kwargs) -> Optional[str]:
    """
    Безопасно извлекает текст первого найденного элемента в поддереве.

    Args:
        parent: BeautifulSoup-объект или элемент DOM.
        tag: Искомый HTML-тег.
        **kwargs: Дополнительные аргументы для фильтрации (атрибуты, классы).

    Returns:
        Optional[str]: Очищенный текст элемента или None, если не найден.
    """
    element = parent.find(tag, kwargs)
    return element.get_text(strip=True) if element else None


def get_next_page_url(html: str, current_url: str) -> Union[str, None]:
    """
    Определяет абсолютный URL следующей страницы пагинации.

    Args:
        html: HTML-код текущей страницы выдачи.
        current_url: URL текущей страницы для разрешения относительных путей.

    Returns:
        Union[str, None]: Ссылка на следующую страницу или None.
    """
    soup = BeautifulSoup(html, 'html.parser')
    next_btn = soup.find("a", {"data-ftid": "component_pagination-item-next"})
    if next_btn and next_btn.get("href"):
        return urljoin(current_url, next_btn["href"])
    return None


def parse_listing_pages(html: str, page_url: str) -> List[dict]:
    """
    Извлекает массив превью-данных объявлений со страницы выдачи.

    Парсит блоки объявлений, формирует корректные абсолютные URL,
    извлекает ad_id из адреса и фиксирует позицию в поиске.

    Args:
        html: HTML-код страницы листинга.
        page_url: URL текущей страницы (используется для referer и urljoin).

    Returns:
        List[dict]: Список словарей с полями ad_id, url, price_in_list,
        header_in_list, position_in_search, referer и датой парсинга.
    """
    soup = BeautifulSoup(html, 'html.parser')
    items = soup.find_all("div", {"data-ftid": "bulls-list_bull"})
    ads = []

    for idx, item in enumerate(items):
        link = item.find("a", href=True)
        if not link:
            continue

        ad_url = link["href"]
        if not ad_url.startswith("http"):
            ad_url = f"https://auto.drom.ru{ad_url}"
        ad_id = re.search(r"\d{7,}", ad_url)

        ads.append({
            "ad_id": str(ad_id.group()) if ad_id else None,
            "url": ad_url,
            "header_in_list": safe_get_text(item, 'a', **{"data-ftid": "bull_title"}),
            "price_in_list": safe_get_text(item, 'span', **{"data-ftid": "bull_price"}),
            "list_parse_date": datetime.date.today().isoformat(),
            "position_in_search": idx,
            "referer": page_url
        })
    return ads
