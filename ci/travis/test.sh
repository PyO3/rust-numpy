#!/bin/sh

set -ex

cargo build --verbose --features $FEATURES
cargo test --verbose --features $FEATURES -- --test-threads=1
rustdoc -L target/debug/deps/ --test README.md

for example in examples/*; do
    if [ $example != 'examples/linalg' ]; then
        cd $example
        tox -e py
        cd $TRAVIS_BUILD_DIR
    fi
done
