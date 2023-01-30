"""
Author: Karthikraja

Instagram message sending Program 
"""
from instabot import Bot
bot = Bot()

bot.login(username="Username", password="Password")
bot.send_message("Hi Guys", ["Receiver's Username"])