#!/bin/sh

set -ex

cargo build --verbose --features $FEATURES
cargo test --verbose --features $FEATURES
rustdoc -L target/debug/deps/ --test README.md

for example in examples/*; do
  cd $example
  tox -e py
  cd $TRAVIS_BUILD_DIR
done
