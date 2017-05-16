#!/usr/bin/env python

import rust_ext
import numpy as np

x = np.array([1.0, 2.0])
y = np.array([2.0, 3.0])
result = rust_ext.axpy(3, x, y)
print(result)
