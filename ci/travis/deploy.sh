#!/bin/sh

cargo doc --all-features --no-deps &&
echo "<meta http-equiv=refresh content=0;url=numpy/index.html>" > target/doc/index.html &&
git clone https://github.com/davisp/ghp-import.git &&
./ghp-import/ghp_import.py -n \
                           -p \
                           -f \
                           -m "Documentation upload" \
                           -r https://"$GH_TOKEN"@github.com/"$TRAVIS_REPO_SLUG.git" \
                           target/doc &&
echo "Uploaded documentation"

