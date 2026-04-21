from bs4 import BeautifulSoup
import datetime
import re


def parse_ads(html, ad_id):
    soup = BeautifulSoup(html, "html.parser")

    def get_text(selector):
        el = soup.select_one(selector)
        return el.get_text(" ", strip=True) if el else None

    def extract_registrations():
        vin_block = soup.select_one("[data-ga-stats-name='gibdd_report']")
        if not vin_block:
            return None

        text = vin_block.get_text(" ", strip=True).lower()

        match = re.search(r"(\d+)\s+запис[а-я]+\s+о\s+регистрац", text)
        if match:
            return int(match.group(1))

        return None

    def parse_seller_info():
        # 1. Проверяем, является ли дилером
        dealer_block = soup.select_one("[data-ftid='bulletin-card_dealer']")
        of_dealer_block = soup.select_one("[data-ftid='bulletin-card_dealer-official']")
        if dealer_block or of_dealer_block:
            return "dealer"

        # 2. Частное лицо / собственник
        title_el = soup.select_one("[data-ftid='bulletin-card_title']")
        if title_el:
            title_text = title_el.get_text(strip=True).lower()

            if "частное лицо" in title_text:
                return "private"

            if "собственник" in title_text:
                return "owner"

        return None

    data = {
        "ad_id": ad_id,
        "parse_date": datetime.date.today(),
        "header": get_text("h1[data-ftid='page-title']"), #внутри марка, модель, год
        "city": get_text("div[data-ftid='city'] span[data-ftid='value']"),
        "price": get_text("div[data-ftid='bulletin-price']"),
        "mileage_km": get_text("tr[data-ftid='specification-mileage'] td[data-ftid='value']"),
        "engine_vol": get_text("tr[data-ftid='specification-engine'] td[data-ftid='value']"),# тип, объем
        "engine_hp": get_text("tr[data-ftid='specification-power'] td[data-ftid='value']"),
        "transmission_type": get_text("tr[data-ftid='specification-transmission'] td[data-ftid='value']"),
        "drive_type": get_text("tr[data-ftid='specification-drive'] td[data-ftid='value']"),
        "color": get_text("tr[data-ftid='specification-color'] td[data-ftid='value']"),
        "num_owners_from_specs": get_text("tr[data-ftid='specification-owners'] td[data-ftid='value']"),
        "num_owners_from_vin": extract_registrations(),
        "wheel_side": get_text("tr[data-ftid='specification-wheel'] td[data-ftid='value']"),
        "generation": get_text("tr[data-ftid='specification-generation'] td[data-ftid='value']"),
        "complectation": get_text("tr[data-ftid='specification-complectation'] td[data-ftid='value']"),
        "seller_type": parse_seller_info(),
        "seller_reg_time": get_text("div[data-ftid='bulletin-card_age']"), # Более N лет на Дроме
        "drom_price_cat": get_text("div[data-ga-stats-name='good_deal_mark']"),
        "raw_text": get_text("div[data-ftid='info-full'] span[data-ftid='value']")
                    or get_text("div[data-ftid='info-short'] span[data-ftid='value']")
    }

    return data
