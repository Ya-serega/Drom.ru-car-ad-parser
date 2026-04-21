import requests
import time
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PageFetcher:
    def __init__(self, headers: dict, timeout: int = 15, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(headers)

    def get(self, url: str, referer: Optional[str] = None) -> Optional[str]:
        headers = {}
        if referer:
            headers["Referer"] = referer

        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                status = response.status_code

                if status == 404:
                    logger.warning(f"404 Not Found: {url}")
                    return None
                if status == 403:
                    logger.warning(f"403 Forbidden: {url}")
                    return None
                if status == 429:
                    wait = random.uniform(60, 180)
                    logger.info(f"429 Rate Limit. Waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue
                if 500 <= status < 600:
                    wait = random.uniform(15, 30)
                    logger.warning(f"Server error {status}. Retry {attempt}/{self.retries}")
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.text

            except requests.exceptions.Timeout:
                wait = random.uniform(30, 60)
                logger.warning(f"Timeout {attempt}/{self.retries} for {url}")
                time.sleep(wait)
            except requests.RequestException as e:
                wait = random.uniform(40, 60)
                logger.warning(f"Request error {attempt}/{self.retries} for {url}: {e}")
                time.sleep(wait)

        logger.error(f"Failed after {self.retries} retries: {url}")
        return None
