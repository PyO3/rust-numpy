#!/bin/sh

set -ex

cargo build --verbose
cargo test --verbose

for example_dir in examples/*; do
    if [ $example != 'examples/linalg' ]; then
        pushd $example_dir
        tox -c "tox.ini" -e py
        popd
    fi
done
