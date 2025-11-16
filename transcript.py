"""
YouTube Transcript Scraper using Selenium
"""

import time
import random
import pandas as pd
from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchElementException, TimeoutException
)


# =========================
# CONFIGURATION
# =========================
INPUT_CSV = "DataSet/Scraped Videos/Scraped2025-11-15.csv"
OUTPUT_CSV = "video_transcripts.csv"
MAX_VIDEOS = 500
CHUNK_SIZE = 3  # save every X transcripts
MIN_WAIT, MAX_WAIT = 3, 6
SCRAPE_URL = "https://youtubetotranscript.com/"


# =========================
# SELENIUM SCRAPER CLASS
# =========================
class TranscriptScraper:
    def __init__(self):
        # Make sure chromedriver is installed and added to PATH
        self.driver = webdriver.Chrome()

    def close_popups(self):
        try:
            popup = self.driver.find_element(By.CSS_SELECTOR, ".popup, .modal, .overlay")
            close_btn = popup.find_element(By.CSS_SELECTOR, ".close, .btn-close, .close-button")
            close_btn.click()
            time.sleep(1)
        except NoSuchElementException:
            pass

    def fetch_once(self, video_url):
        """Try once to fetch transcript"""
        try:
            self.driver.get(SCRAPE_URL)
            self.close_popups()

            input_box = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "youtube_url"))
            )
            input_box.clear()
            input_box.send_keys(video_url)

            submit_btn = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            submit_btn.click()

            transcript_container = WebDriverWait(self.driver, 25).until(
                EC.presence_of_element_located((By.ID, "transcript"))
            )

            segments = transcript_container.find_elements(By.CSS_SELECTOR, "span.transcript-segment")
            transcript_text = " ".join(seg.text for seg in segments if seg.text.strip())

            return transcript_text if transcript_text.strip() else None

        except TimeoutException:
            print(f"Timeout while loading transcript for {video_url}")
            return None

        except Exception as e:
            print(f"Error for {video_url}: {e}")
            return None

    def get_transcript(self, video_url, retries=1):
        """Retry once if failed"""
        transcript = self.fetch_once(video_url)
        if transcript is None and retries > 0:
            print(f"Retrying {video_url} after reload...")
            time.sleep(3)
            self.driver.refresh()
            return self.get_transcript(video_url, retries - 1)
        return transcript

    def quit(self):
        self.driver.quit()


# =========================
# MAIN WORKFLOW
# =========================
def main():

    scraper = TranscriptScraper()

    # Load CSV
    data = pd.read_csv(INPUT_CSV, encoding="utf-8")

    # Filter out Shorts
    data_filtered = data[~data['title'].str.contains("#", case=False, na=False)]
    data_filtered = data_filtered[data_filtered['duration_seconds'] >= 120]

    # Sort by views descending
    # data_sorted = data_filtered.sort_values(by='views', ascending=False).sort_values(by='duration_seconds', ascending=False)
    # Take top 40
    # top40_videos = data_sorted.drop_duplicates(subset=['url']).head(40)
    # print(f"Processing {len(top40_videos)} videos for transcripts...")

    # Get URLs as a list
    # video_urls = top40_videos['url'].tolist()

    # Optional: also get titles for reference
    # video_titles = top40_videos['title'].tolist()

    video_urls = data_filtered['url'][:MAX_VIDEOS].tolist()

    # Load existing transcripts if available
    try:
        transcripts = pd.read_csv(OUTPUT_CSV)
        existing_urls = set(transcripts["video_url"])
        print(f"Loaded existing transcripts: {len(existing_urls)}")
    except FileNotFoundError:
        transcripts = pd.DataFrame(columns=["video_url", "transcript"])
        existing_urls = set()

    processed = 0

    # tqdm progress bar
    for idx, video_url in enumerate(tqdm(video_urls, desc="Processing videos"), start=1):

        if video_url in existing_urls:
            print(f"[{idx}/{len(video_urls)}] Skipping already processed {video_url}")
            continue

        transcript = scraper.get_transcript(video_url)
        if transcript is None:
            print(f"[{idx}/{len(video_urls)}] Failed, skipping {video_url}")
            continue

        transcripts.loc[len(transcripts)] = {"video_url": video_url, "transcript": transcript}
        print(f"[{idx}/{len(video_urls)}] Done: {video_url} ({len(transcript)} chars)")

        processed += 1
        if processed % CHUNK_SIZE == 0:
            transcripts.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
            print(f"Saved progress after {processed} new transcripts.")

        # Random delay to mimic human behavior
        time.sleep(random.uniform(MIN_WAIT, MAX_WAIT))

    # Save any remaining transcripts
    transcripts.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"Finished. Total transcripts saved: {len(transcripts)}")

    scraper.quit()


if __name__ == "__main__":
    main()
