#!/bin/sh

set -ex

cargo build --verbose
cargo test --verbose

if [[ $RUN_LINT == 1 ]]; then
    cargo fmt --all -- --check
    cargo clippy --tests
    for example in examples/*; do (cd $$example/; cargo clippy) || exit 1; done
fi

for example in examples/*; do
    if [ $example != 'examples/linalg' ]; then
        cd $example
        tox -e py
        cd $TRAVIS_BUILD_DIR
    fi
done
