#!/usr/bin/python

from target.debug import librust2py

arr = librust2py.get_arr()
for i in arr:
    print(type(i))
