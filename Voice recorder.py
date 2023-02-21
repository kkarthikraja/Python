"""
Author: Karthikraja

Voice recorder Program 
"""
from googleapiclient.discovery import build
import datetime

API_KEY = "API key"
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_most_viewed(region_code, date):
    request = youtube.videos().list(
        part='snippet,statistics',
        chart='mostPopular',
        regionCode=region_code,
        maxResults=50,
        publishedAfter=date
    )
    response = request.execute()
    return response

region_code = 'US'
date = datetime.datetime(2022, 1, 30).isoformat() + 'Z'
most_viewed = get_most_viewed(region_code, date)

video_list = []
for item in most_viewed['items']:
    video_data = {
        'title': item['snippet']['title'],
        'views': item['statistics']['viewCount']
    }
    video_list.append(video_data)

