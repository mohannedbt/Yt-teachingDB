from googleapiclient.discovery import build
import pandas as pd
import time
import isodate
import os
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
API_KEY = os.getenv("API_KEY")
SEARCH_TERMS = ["Linear algebra easy", "linear algebra for beginners", "linear algebra explained in under 10 min", "full dive in linear algebra"]
MAX_VIDEOS_PER_TERM = 200  # limit to stay within quota
BATCH_SIZE = 50  # max videos per videos.list call
CSV_FILE = "Linear_algebra_yt_vids.csv"

# -----------------------------
# Initialize YouTube API client
# -----------------------------
youtube = build("youtube", "v3", developerKey=API_KEY)

# -----------------------------
# Helper function to fetch video details
# -----------------------------
def fetch_video_details(video_ids):
    response = youtube.videos().list(
        part="snippet,contentDetails,topicDetails,statistics",
        id=",".join(video_ids)
    ).execute()
    
    videos_info = []
    for item in response.get("items", []):
        vid = item["id"]
        snippet = item["snippet"]
        stats = item.get("statistics", {})
        duration = item["contentDetails"].get("duration", "N/A")
        duration_sec = isodate.parse_duration(duration).total_seconds() if duration != "N/A" else None
        topics = item.get("topicDetails", {}).get("topicCategories", [])
        views = stats.get("viewCount", 0)
        
        videos_info.append({
            "search_term": "",  # will set later
            "video_id": vid,
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "duration": duration,
            "duration_seconds": duration_sec,
            "views": views,
            "topics": ",".join(topics),
            "url": f"https://www.youtube.com/watch?v={vid}"
        })
    return videos_info


# -----------------------------
# Main scraping loop
# -----------------------------
all_videos = []

for term in SEARCH_TERMS:
    print(f"Searching for '{term}'...")
    next_page_token = None
    total_fetched = 0
    
    while total_fetched < MAX_VIDEOS_PER_TERM:
        remaining = MAX_VIDEOS_PER_TERM - total_fetched
        fetch_count = min(50, remaining)
        search_response = youtube.search().list(
            q=term,
            part="id,snippet",
            type="video",
            maxResults=fetch_count,
            pageToken=next_page_token
        ).execute()
        
        video_ids = [item["id"]["videoId"] for item in search_response["items"]]
        
        # Batch fetch details
        for i in range(0, len(video_ids), BATCH_SIZE):
            batch_ids = video_ids[i:i+BATCH_SIZE]
            video_details = fetch_video_details(batch_ids)
            
            # Assign search term and append to list
            for vid in video_details:
                vid["search_term"] = term
                all_videos.append(vid)
            
            total_fetched += len(batch_ids)
        
        next_page_token = search_response.get("nextPageToken")
        if not next_page_token:
            break
        
        time.sleep(1)  # avoid hammering API
    
    print(f"Done fetching up to {total_fetched} videos for '{term}'.")

# -----------------------------
# Save to CSV
# -----------------------------
df = pd.DataFrame(all_videos)
df.to_csv(CSV_FILE, index=False)
print(f"All data saved to {CSV_FILE}")
