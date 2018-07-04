#!/bin/sh

set -ex

cargo build --verbose --all
cargo test --verbose --all

cd example
pip install -r requirements.txt
python setup.py develop
./test.py
