#!/usr/bin/env python3
import sys
import json

"""
Multiplies first N integers (incl.),
optionally skipping multiples of k,
or squaring the numbers before summing
"""

with open(sys.argv[1]) as f:
    config = json.load(f)

tot = 1
for i in range(1, config['N']+1):
    if 'k' in config is not None and i%config['k'] == 0:
        continue
    if config.get('square', False):
        i = i**2
    tot *= i

print(tot)