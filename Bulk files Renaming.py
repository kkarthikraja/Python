"""
Author: Karthikraja

Bulk file Renaming Program
"""
import os

def main():
    i = 0
    path = "./Python file/"
    for filename in os.listdir(path):
        file_dest = "image" + str(i) + ".jpg"
        file_source = path + filepath
        file_dest = path + file_path
        os.rename(my_source, my_dest)
        i += 1
        
if __name__ == '__main__':
    main()