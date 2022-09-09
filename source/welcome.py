# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:07:05 2021

@author: Sivasangaran V
"""

import cv2
import argparse
def parse_args():
    parser=argparse.ArgumentParser(description=("Set up classifier model"))
    parser.add_argument('-img',help="Image file path")
    return parser.parse_args()
args=parse_args()
img = cv2.imread(args.img)
while True:
    cv2.imshow('Welcome',img)
    k = cv2.waitKey(30) & 0xFF
    if k==27:
        break
cv2.destroyAllWindows()