
from setuptools import setup
from setuptools_rust import RustExtension, Binding

setup(name='rust_ext',
      version='1.0',
      rust_extensions=[
          RustExtension('rust_ext._rust_ext', 'extensions/Cargo.toml',
                        binding=Binding.RustCPython)],
      packages=['rust_ext'],
      zip_safe=False)
