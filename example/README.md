Example of rust-numpy
----------------------

- Prerequirement

```
sudo pip3 install -r requirements.txt
```

- Build

```
python3 setup.py develop --user
```

- Run

```python
import numpy as np
import rust_ext

a = np.array([0.0, 1.0])
b = np.array([2.0, 3.0])
rust_ext(2.0, a, b)
```
