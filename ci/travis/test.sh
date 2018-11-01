#!/bin/sh

set -ex

cargo build --verbose --all
cargo test --verbose --all
rustdoc -L target/debug/deps/ --test README.md

for example in examples/*; do
  cd $example
  tox -e py
  cd $TRAVIS_BUILD_DIR
done
