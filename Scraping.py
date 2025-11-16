from googleapiclient.discovery import build
import pandas as pd
import time
import isodate
import os
from datetime import date as Date
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------
# Configuration
# -------------------------------------
API_KEY = os.getenv("API_KEY2")
MAX_VIDEOS_PER_TERM = 200
BATCH_SIZE = 50
CSV_FILE = "ScrapedVideos"+str(Date.today())+".csv"

SEARCH_TERMS = [
    "derivatives explained easy",
    "complexe analysis for beginners",
    "complexe numbers explained in under 10 min",
    "full dive in Complexe analysis",
]

# -------------------------------------
# YouTube API Client Init
# -------------------------------------
def get_youtube_client():
    return build("youtube", "v3", developerKey=API_KEY)


# -------------------------------------
# Fetch video details by batch of IDs
# -------------------------------------
def fetch_video_details(youtube, video_ids):
    response = youtube.videos().list(
        part="snippet,contentDetails,topicDetails,statistics",
        id=",".join(video_ids)
    ).execute()
    
    videos_info = []
    for item in response.get("items", []):
        vid = item["id"]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        duration_iso = item["contentDetails"].get("duration", "PT0S")
        duration_sec = isodate.parse_duration(duration_iso).total_seconds()
        
        # Skip shorts < 120s
        if duration_sec < 120:
            continue
        
        videos_info.append({
            "video_id": vid,
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "duration": duration_iso,
            "duration_seconds": duration_sec,
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
            "topics": ",".join(item.get("topicDetails", {}).get("topicCategories", [])),
            "url": f"https://www.youtube.com/watch?v={vid}"
        })
        
    return videos_info


# -------------------------------------
# Search using query term
# -------------------------------------
def search_videos_by_term(youtube, term, max_results):
    all_videos = []
    total_fetched = 0
    next_page_token = None
    
    print(f"\nSearching for '{term}'...")

    while total_fetched < max_results:
        remaining = max_results - total_fetched
        fetch = min(50, remaining)

        response = youtube.search().list(
            q=term,
            part="id,snippet",
            type="video",
            maxResults=fetch,
            pageToken=next_page_token
        ).execute()

        video_ids = [item["id"]["videoId"] for item in response["items"]]

        # Batch details lookup
        for i in range(0, len(video_ids), BATCH_SIZE):
            batch_ids = video_ids[i:i+BATCH_SIZE]
            details = fetch_video_details(youtube, batch_ids)

            for vid in details:
                vid["search_term"] = term
                all_videos.append(vid)

            total_fetched += len(batch_ids)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(1)

    print(f"Fetched {len(all_videos)} videos for '{term}'")
    return all_videos


# -------------------------------------
# Main controller
# -------------------------------------
def scrape_all_terms(terms):
    youtube = get_youtube_client()
    all_data = []

    for term in terms:
        all_data.extend(search_videos_by_term(youtube, term, MAX_VIDEOS_PER_TERM))

    return all_data


# -------------------------------------
# Save results
# -------------------------------------
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nSaved {len(df)} videos -> {filename}")


# -------------------------------------
# Run script
# -------------------------------------
if __name__ == "__main__":
    data = scrape_all_terms(SEARCH_TERMS)
    save_to_csv(data, CSV_FILE)
