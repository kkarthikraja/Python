"""
Author: Karthikraja

Web Scraping Program
"""
import requests 
from bs4 import BeautifulSoup as bs

github_usr = input('Input Github User: ')
url = 'https://github.com/'+github_usgithub_usr 
r = requests.get(url)
soup = bs(r.content, 'html.parser')
profile_img = soup.find('img', {'alt' : 'Avatar'})['src']
print(profile_img)