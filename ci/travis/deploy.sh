#!/bin/sh

if [["$TRAVIS_PULL_REQUEST" = "false" && "$TRAVIS_BRANCH" == "master" && \
         "$TRAVIS_RUST_VERSION" == "nightly" ]]; then
    cargo doc --all-features --no-deps &&
    echo "<meta http-equiv=refresh content=0;url=os_balloon/index.html>" > target/doc/index.html &&
    git clone https://github.com/davisp/ghp-import.git &&
    ./ghp-import/ghp_import.py -n \
                               -p \
                               -f \
                               -m "Documentation upload" \
                               -r https://"$GH_TOKEN"@github.com/"$TRAVIS_REPO_SLUG.git" \
                               target/doc &&
    echo "Uploaded documentation"
fi
