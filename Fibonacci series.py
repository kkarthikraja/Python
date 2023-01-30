"""
Author: Karthikraja

Fibonacci code Program using recursive and not
"""
def getNthfib(n):
    if n ==1:
        return 0
    if n <= 3:
        return 1
    
    return getNthfib(n-1) + getNthfib(n-2)

def getNthfib(n, calculated = {1:0, 2:1. 3:1}):
    if n in calculated:
        return calculated[n]
    
    calculated[n] = getNthfib(n-1, calculated) + getNthfib(n-2, calculated)
    return calculated[n]