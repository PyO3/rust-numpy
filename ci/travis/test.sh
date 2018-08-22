#!/bin/sh

set -ex

cargo build --verbose --all
cargo test --verbose --all
rustdoc -L target/debug/deps/ --test README.md

cd example
python setup.py install
python setup.py test
