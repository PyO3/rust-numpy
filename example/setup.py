
from setuptools import setup
from setuptools_rust import RustExtension

setup(name='rust_ext',
      version='1.0',
      rust_extensions=[
          RustExtension('rust_ext._rust_ext', 'extensions/Cargo.toml')],
      packages=['rust_ext'],
      zip_safe=False)
