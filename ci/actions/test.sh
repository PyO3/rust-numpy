#!/bin/sh

set -ex

cargo build --verbose
cargo test --verbose -- --test-threads=1

for example_dir in examples/*; do
    cd $example_dir
    tox -c "tox.ini" -e py
    cd -
done
