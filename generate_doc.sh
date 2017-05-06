#!/bin/bash
set -e
[[ "$(git symbolic-ref --short HEAD)" == "master" ]] || exit 0

msg() {
    echo "[1;34m> [1;32m$@[0m"
}

dir="$(pwd)"
tmp="$(mktemp -d)"
last_rev="$(git rev-parse HEAD)"
last_msg="$(git log -1 --pretty=%B)"

trap "cd \"$dir\"; rm -rf \"$tmp\"" EXIT

msg "Cloning into a temporary directory..."
git clone -qb gh-pages $dir $tmp
cd "$tmp"
git checkout -q master
ln -s $dir/target $tmp/target

msg "Generating documentation..."
cargo doc

# Switch to pages
msg "Replacing documentation..."
git checkout -q gh-pages

# Clean and replace
git rm -q --ignore-unmatch -rf .
git reset -q -- .gitignore
git checkout -q -- .gitignore
cp -a target/doc/* .
rm target
git add .
git commit -m "Update docs for $last_rev" -m "$last_msg"
git push -qu origin gh-pages
msg "Done."
