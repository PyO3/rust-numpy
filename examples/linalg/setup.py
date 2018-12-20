import sys
from setuptools import find_packages, setup
from setuptools_rust import RustExtension


PYTHON_MAJOR_VERSION = sys.version_info[0]

setup_requires = ['setuptools-rust>=0.6.0']
install_requires = ['numpy']
test_requires = install_requires + ['pytest']

setup(
    name='rust_linalg',
    version='0.1.0',
    description='Example of python extension with ndarray-linalg',
    rust_extensions=[RustExtension(
        'rust_linalg.rust_linalg',
        './Cargo.toml',
        rustc_flags=['--cfg=Py_{}'.format(PYTHON_MAJOR_VERSION)],
        features=['numpy/python{}'.format(PYTHON_MAJOR_VERSION)],
    )],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
)
