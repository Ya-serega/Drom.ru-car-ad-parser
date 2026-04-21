"""
Оркестратор пайплайна сбора данных drom.ru.

Управляет многопоточным парсингом листинга и карточек объявлений,
обеспечивает устойчивые чекпоинты сессий, отслеживание истории
(history_ids), graceful shutdown и итоговую сборку датасета.
"""
import os
import time
import json
import logging
import signal
import shutil
import gc
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Union, List

from src.list_parser import parse_listing_pages, get_next_page_url
from src.ads_parser import parse_ads
from src.fetcher import PageFetcher
from src.dataset_builder import build_dataset
import src.config as config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("pipeline.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    Менеджер состояния парсинга и управления чекпоинтами.

    Отвечает за сохранение прогресса листинга, атомарную запись
    распарсенных объявлений, ведение глобальной истории ad_id
    и финализацию сессии с сохранением в Parquet.
    """

    def __init__(self, session_id: str):
        """
        Инициализирует директорию сессии и загружает состояние.

        Args:
            session_id: Уникальный идентификатор сессии (обычно timestamp).
        """
        self.id = session_id
        self.dir = Path(config.SESSIONS_DIR) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.ckpt_listings = self.dir / "listings_state.json"
        self.ckpt_ads = self.dir / "ads_parsed.csv"
        self.history_file = Path(config.HISTORY_IDS_PATH)

        # История всех когда-либо спарсенных ID
        self.history_ids: set[str] = set()
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                self.history_ids.update(line.strip() for line in f if line.strip())
        logger.info(f"History loaded: {len(self.history_ids)} unique ad_ids")

        # Состояние листинга
        self.listing_state = {}
        if self.ckpt_listings.exists():
            with open(self.ckpt_listings, "r", encoding="utf-8") as f:
                self.listing_state = json.load(f)

        # Уже спарсенные ID в текущей сессии (для resume)
        self.session_ids: set[str] = set()
        if self.ckpt_ads.exists():
            try:
                df = pd.read_csv(self.ckpt_ads, usecols=["ad_id"], dtype={"ad_id": str})
                self.session_ids.update(df["ad_id"].dropna().astype(str))
            except Exception as e:
                logger.warning(f"Failed to load session checkpoint: {e}")

    def save_listing_state(self, state: dict) -> None:
        """
        Атомарно сохраняет состояние пагинации для URL.

        Использует временный файл и os.replace() для гарантии целостности.

        Args:
            state: Словарь с текущей страницей, next_url и списком ad_id.
        """
        self.listing_state.update(state)
        tmp = self.ckpt_listings.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.listing_state, f, ensure_ascii=False)
        os.replace(tmp, self.ckpt_listings)  # Atomic rename

    def append_ads(self, batch: list[dict]) -> None:
        """Добавляет пакет распарсенных объявлений в чекпоинт-файл.

        Args:
            batch: Список словарей с данными объявлений.
        """
        if not batch: return
        df = pd.DataFrame(batch)
        file_exists = self.ckpt_ads.exists()
        df.to_csv(self.ckpt_ads, mode="a", header=not file_exists, index=False)
        self.session_ids.update(df["ad_id"].astype(str))

    def finalize(self, final_df: pd.DataFrame) -> Path:
        """
        Сохраняет итоговый датасет, обновляет историю и очищает сессию.

        Генерирует динамическое имя файла на основе брендов и количества
        строк, сохраняет в Parquet, добавляет новые ad_id в history_ids.txt
        и пытается удалить временные файлы сессии (с обработкой блокировок Windows).

        Args:
            final_df: Финальный DataFrame с признаками.

        Returns:
            Path: Путь к сохраненному файлу .parquet.
        """
        # Динамический нейминг
        brands = final_df["brand"].dropna().unique()
        brand_tag = "_".join(sorted(brands))[:50] if len(brands) <= 5 else "multi_brand"
        count = len(final_df)
        ts = self.id[:15]
        filename = f"session_{ts}_{brand_tag}_{count}.parquet"
        out_path = Path(config.DATA_PROCESSED_DIR) / filename

        final_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(f"Dataset saved: {out_path} ({count} rows)")

        # Обновление истории
        with open(self.history_file, "a", encoding="utf-8") as f:
            new_ids = self.session_ids - self.history_ids
            if new_ids:
                f.write("\n".join(new_ids) + "\n")

        def _is_file_locked(filepath: Path) -> bool:
            """Проверка, заблокирован ли файл (Windows)"""
            try:
                with open(filepath, 'a'):
                    return False
            except PermissionError:
                return True

        # Освобождение ресурсов перед удалением (критично для Windows)
        if self.ckpt_ads.exists() and _is_file_locked(self.ckpt_ads):
            logger.warning(f"File still locked: {self.ckpt_ads}")

        gc.collect()
        time.sleep(0.5)

        # Очистка сессии с обработкой ошибок
        try:
            shutil.rmtree(self.dir)
            logger.info("Session checkpoints cleaned. History updated.")
        except PermissionError as e:
            logger.warning(f"Could not delete session dir (file lock on Windows): {self.dir}")
            logger.warning(f"Error: {e}")
            logger.info("Tip: Close any file explorers or editors viewing these files, or delete manually later.")
        except Exception as e:
            logger.error(f"Unexpected error during cleanup: {e}")

        return out_path


class PipelineOrchestrator:
    """
    Главный контроллер пайплайна сбора данных.

    Координирует обход страниц, многопоточный парсинг карточек,
    дедупликацию, буферизацию записи и вызов Dataset Builder.
    Поддерживает безопасную остановку по SIGINT/SIGTERM.
    """

    def __init__(self, max_workers: int = 4, batch_size: int = 100, max_pages: int = 1000):
        """
        Настраивает пул воркеров, параметры буфера и обработчики сигналов.

        Args:
            max_workers: Количество потоков для парсинга карточек.
            batch_size: Размер пакета перед записью в чекпоинт.
            max_pages: Лимит страниц на один URL листинга.
        """
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_pages = max_pages
        self.fetcher = PageFetcher(config.HEADERS)
        self.session = SessionManager(datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.lock = threading.Lock()
        self.buffer: list[dict] = []
        self.running = True

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        config.init_dirs()
        os.makedirs(config.SESSIONS_DIR, exist_ok=True)

    def _handle_shutdown(self, signum):
        """
        Обработчик сигналов прерывания (Ctrl+C, kill).

        Устанавливает флаг остановки для безопасного завершения цикла.
        """
        logger.warning(f"Shutdown signal {signum} received. Flushing...")
        self.running = False

    def _flush_buffer(self):
        """
        Поточно-безопасная запись буфера в чекпоинт сессии.

        Использует threading.Lock для предотвращения race conditions.
        """

        with self.lock:
            if self.buffer:
                self.session.append_ads(self.buffer)
                logger.info(f"Flushed {len(self.buffer)} ads to checkpoint.")
                self.buffer.clear()

    def collect_listings(self) -> List[dict]:
        """
        Итерирует по TARGET_URLS, собирает превью объявлений с пагинацией.

        Поддерживает resume по сохраненному состоянию листинга.
        Сохраняет прогресс после каждой страницы.

        Returns:
            list[dict]: Список словарей с метаданными объявлений из листинга.
        """
        all_records = []
        for url in config.TARGET_URLS:
            # Resume или старт с начала
            state = self.session.listing_state.get(url, {"page": 0, "next_url": url, "ads": []})
            current_url = state["next_url"]
            page = state["page"]

            logger.info(f"Crawling: {url} | Resume page: {page}")
            while self.running and current_url and page < self.max_pages:
                page += 1
                time.sleep(2.0)
                html = self.fetcher.get(current_url, referer=config.HEADERS.get("Referer"))
                if not html: break

                records = parse_listing_pages(html, current_url)
                if not records: break

                for r in records:
                    r["ad_id"] = str(r["ad_id"])
                    r["referer"] = current_url
                all_records.extend(records)
                logger.info(f"Page {page}: {len(records)} ads found.")

                # Сохраняем чекпоинт
                self.session.save_listing_state({
                    url: {"page": page, "next_url": get_next_page_url(html, current_url) or "",
                          "ads": [r["ad_id"] for r in records]}
                })

                next_url = get_next_page_url(html, current_url)
                if not next_url or next_url == current_url: break
                current_url = next_url
            logger.info(f"Finished: {url}")
        return all_records

    def run(self):
        """
        Запускает полный цикл пайплайна: сбор → фильтрация → парсинг → сборка.

        Выполняет:
        1. Сбор листинга с учетом resume.
        2. Фильтрацию уже известных ad_id (history + session).
        3. Многопоточный парсинг карточек с буферизацией.
        4. Финальную сборку датасета через build_dataset.
        Обеспечивает логирование времени выполнения и graceful exit.
        """
        logger.info(f"=== Pipeline Started | Session: {self.session.id} ===")
        t_start = time.time()

        # 1. Сбор листинга
        list_records = self.collect_listings()
        if not list_records:
            logger.error("No listings found. Exiting.")
            return
        logger.info(f"Total listings: {len(list_records)}")

        # 2. Фильтрация дублей
        ads_to_parse = [r for r in list_records if
                        r["ad_id"] not in self.session.history_ids and r["ad_id"] not in self.session.session_ids]
        logger.info(f"Ads to parse (after dedup): {len(ads_to_parse)}")

        # Ранний выход, если нечего парсить
        if not ads_to_parse:
            logger.info("All ads already in history. Skipping parsing stage.")
            try:
                shutil.rmtree(self.session.dir)
                logger.info("Session cleaned. No new data collected.")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")
            return

        # 3. Параллельный парсинг
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._parse_ad, ad): ad for ad in ads_to_parse}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Parsing ads")

            for future in pbar:
                if not self.running: break
                try:
                    res = future.result()
                    if res:
                        self.buffer.append(res)
                        if len(self.buffer) >= self.batch_size:
                            self._flush_buffer()
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
            self._flush_buffer()

        if not self.running:
            logger.info("Shutdown requested. Session saved for resume.")
            return

        # 4. Финальная сборка
        logger.info("Building dataset...")

        # Чекпоинт может не существовать, если не было новых объявлений для парсинга
        if self.session.ckpt_ads.exists() and self.session.ckpt_ads.stat().st_size > 0:
            df_ads = pd.read_csv(self.session.ckpt_ads, dtype={"ad_id": str}).fillna("")
        else:
            logger.info("No new ads parsed in this session. Exiting gracefully.")
            # Обновляем историю (на случай если были дубли, но не парсили)
            with open(self.session.history_file, "a", encoding="utf-8") as f:
                new_ids = self.session.session_ids - self.session.history_ids
                if new_ids:
                    f.write("\n".join(str(i) for i in new_ids) + "\n")
            # Очищаем сессию
            try:
                shutil.rmtree(self.session.dir)
                logger.info("Session checkpoints cleaned.")
            except Exception as e:
                logger.warning(f"Could not clean session dir: {e}")
            return

        df_list = pd.DataFrame(list_records)
        df_final = build_dataset(df_ads, df_list)
        self.session.finalize(df_final)
        logger.info(f"Total runtime: {time.time() - t_start:.1f} sec")

    def _parse_ad(self, ad: dict) -> Union[dict, None]:
        """
        Воркер-функция для парсинга одной карточки объявления.

        Args:
            ad: Словарь с метаданными (url, ad_id, referer).

        Returns:
            Union[dict, None]: Распарсенные данные карточки или None при ошибке.
        """
        html = self.fetcher.get(ad["url"], referer=ad.get("referer"))
        if not html: return None
        data = parse_ads(html, ad["ad_id"])
        data["ad_id"] = str(data["ad_id"])
        return data


if __name__ == "__main__":
    pipeline = PipelineOrchestrator(max_workers=4, batch_size=50, max_pages = 5)
    pipeline.run()
