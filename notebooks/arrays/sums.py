#!/usr/bin/env python3
import os
import math
import argparse

parser = argparse.ArgumentParser(
    description="Sum first N integers (incl.), "
                "optionally skipping multiples of k, "
                "or squaring the numbers before summing"
)
parser.add_argument('N', type=int)
parser.add_argument('-k', type=int, default=None)
parser.add_argument('-s', '--square', action='store_true')
args = parser.parse_args()

tot = 0
for i in range(1, args.N+1):
    if args.k is not None and i%args.k == 0: 
        continue
    if args.square:
        i = i**2
    tot += i

print(tot)
