#!/bin/sh

set -ex

cargo build --verbose --features $FEATURES
cargo test --verbose --features $FEATURES -- --test-threads=1

for example_dir in examples/*; do
    cd $example_dir
    tox -c "tox.ini" -e py
    cd -
done
