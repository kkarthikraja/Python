"""
Author: Karthikraja

Read Weather info Program
"""
import requests 
from pprint import pprint

API_key = 'cb771e45ac79a4e8e2205c0ce66ff633'

city = input("Enter a city: ")

base_url = "http://api.openweathermap.org/data/2.5/weather?appid="+API_key

weather_dat = requests.get(base_url).json()

pprint(weather_dat)