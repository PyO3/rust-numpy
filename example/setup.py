
import os
import subprocess
import sys
from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand
from setuptools_rust import RustExtension, Binding


class CmdTest(TestCommand):
    def run(self):
        self.run_command("test_rust")
        test_files = os.listdir('./tests')
        ok = 0
        for f in test_files:
            _, ext = os.path.splitext(f)
            if ext == '.py':
                res = subprocess.call([sys.executable, f], cwd='./tests')
                ok = ok | res
        sys.exit(res)


setup_requires = ['setuptools-rust>=0.6.0']
install_requires = ['numpy']
test_requires = install_requires

setup(
    name='rust_ext',
    version='0.1.0',
    description='Example of python-extension using rust-numpy',
    rust_extensions=[RustExtension('rust_ext.rust_ext', 'extensions/Cargo.toml')],
    install_requires=install_requires,
    setup_requires=setup_requires,
    test_requires=test_requires,
    packages=find_packages(),
    zip_safe=False,
    cmdclass=dict(test=CmdTest)
)
