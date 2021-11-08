#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 11:44:43 2021

@author: s174483
"""

import sys

def main():    
    argin = sys.argv
    if len(argin) > 1:
        filename = 'test'+str(argin[1])+'.txt'
    else:
        filename = 'test.txt'
        
    f = open(filename,"w")
    f.write(str(argin))
    f.close()
    return


if __name__ == "__main__":
    main()
    
