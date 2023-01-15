"""
Author: Karthikraja

Password generator Program
"""
import random 

print('Welcome to password generator')

chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*().,:'

num = input('Amount of passwords to generate: ')
num = int(num)

len = input('Input password length: ')
len = int(len)

print('\nhere are the passwords:')

for pwd in range(number):
    passwords = ''
    for c in range(len): 
        passwords += random.choice(chars)