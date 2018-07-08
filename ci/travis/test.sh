#!/bin/sh

set -ex

cargo build --verbose --all
cargo test --verbose --all

cd example
python setup.py install
python setup.py test
