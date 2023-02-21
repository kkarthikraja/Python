"""
Author: Karthikraja

lyrics_extractor Program 
"""
from lyrics_extractor import SongLyrics 

extract_lyrics = SongLyrics(API_KEY, GCS_ENGINE_ID)

extract_lyrics.get_lyrics("Let me down")
