import re
import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Union


def safe_get_text(parent, tag, **kwargs):
    element = parent.find(tag, kwargs)
    return element.get_text(strip=True) if element else None


def get_next_page_url(html: str, current_url: str) -> Union[str, None]:
    soup = BeautifulSoup(html, 'html.parser')
    next_btn = soup.find("a", {"data-ftid": "component_pagination-item-next"})
    if next_btn and next_btn.get("href"):
        return urljoin(current_url, next_btn["href"])
    return None


def parse_listing_pages(html: str, page_url: str) -> list[dict]:
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
