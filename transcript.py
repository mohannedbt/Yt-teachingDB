from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

import pyperclip  # to read from clipboard
import time
import pandas as pd
import random
# Make sure you have chromedriver installed and pyperclip: pip install pyperclip
driver = webdriver.Chrome()

def close_popups(driver):
    try:
        # Example: find popups by common classes or IDs
        popup = driver.find_element(By.CSS_SELECTOR, ".popup, .modal, .overlay")  # adjust selector
        close_btn = popup.find_element(By.CSS_SELECTOR, ".close, .btn-close, .close-button")
        close_btn.click()
        print("Popup closed.")
        time.sleep(1)  # small delay after closing
    except NoSuchElementException:
        # No popup found
        pass

def get_transcript(video_url):
    driver.get("https://youtubetotranscript.com/")
    
    # Step 1: Enter the video URL
    input_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "youtube_url"))
    )
    input_box.clear()
    input_box.send_keys(video_url)
    
    # Step 2: Click submit
    submit_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
    submit_btn.click()
    
    # Step 3: Wait for the transcript container
    transcript_container = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.ID, "transcript"))
    )
    
    # Step 4: Collect all segments
    segments = transcript_container.find_elements(By.CSS_SELECTOR, "span.transcript-segment")
    transcript_text = " ".join([seg.text for seg in segments])
    
    return transcript_text


# Read video URLs from CSV and process
video_urls = pd.read_csv("Linear_algebra_yt_vids.csv")["url"][:12].tolist()
transcripts = []
CHUNK_SIZE = 3  # save every 3 videos

for idx, video_url in enumerate(video_urls, start=1):
    transcript = get_transcript(video_url)
    print(f"[{idx}/{len(video_urls)}] Processed: {video_url}, length: {len(transcript)}")
    
    transcripts.append({
        "video_url": video_url,
        "transcript": transcript
    })
    
    # Save in chunks
    if idx % CHUNK_SIZE == 0:
        df = pd.DataFrame(transcripts)
        df.to_csv("video_transcripts.csv", index=False)
        print(f"Saved {idx} transcripts so far...")
    
    # Random delay to mimic human behavior
    time.sleep(random.uniform(3, 5))

# Save any remaining transcripts
if len(transcripts) % CHUNK_SIZE != 0:
    df = pd.DataFrame(transcripts)
    df.to_csv("video_transcripts.csv", index=False)
    print(f"Saved final transcripts. Total: {len(transcripts)}")

driver.quit()
