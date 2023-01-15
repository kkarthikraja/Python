"""
Author: Karthikraja

Countdown Timer Program
"""
import time

def countdown(s):
    while s:
        mins, secs = divmod(s, 60)
        timer = '{:02}:{02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        s -= 1
    print('Timer completed!')
    
s = input('Enter time in seconds: ')

countdown(int(s))