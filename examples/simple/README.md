# simple extension

Here, we use [`maturin`][maturin] for building Python wheels and
[`nox`][nox] for managing Python dependencies and virtualenvs.

Running `nox` inside this directory creates a virtualenv,
installs Python dependencies and the extension into it
and executes the tests from `tests/test_exp.py`.

By running
```bash
maturin develop
```
from inside a virtualenv, you can use the extension from
the Python REPL:

```python
>>> import numpy as np
>>> import rust_ext
>>> rust_ext.axpy(2.0, np.array([0.0, 1.0]), np.array([2.0, 3.0]))
array([2., 5.])
```

[maturin]: https://github.com/PyO3/maturin
[nox]: https://github.com/theacodes/nox
