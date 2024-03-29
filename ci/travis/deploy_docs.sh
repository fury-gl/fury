#!/bin/bash
if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "master" &&
      $BUILD_DOCS == 1 && $DEPLOY_DOCS == 1 ]]
then
    # See https://help.github.com/articles/creating-an-access-token-for-command-line-use/ for how to generate a token
    # See http://docs.travis-ci.com/user/encryption-keys/ for how to generate
    # a secure variable on Travis
    echo "-- pushing docs --"

    (
    # Setup environment
    export PATH=${ENV_DIR}/miniconda/bin:$PATH
    hash -r
    conda activate testenv
    cd ${TRAVIS_BUILD_DIR}

    # Setup git user
    git config --global user.email "travis@travis-ci.com"
    git config --global user.name "Travis Bot"

    # build docs a second time
    # (cd docs && xvfb-run --server-args="-screen 0, 1920x1080x24" make -C . html)
    (cd docs && make -C . html)

    git clone --quiet --branch=gh-pages https://${GH_REF} doc_build
    cd doc_build

    git rm -r dev
    cp -r ../docs/build/html dev
    git add dev

    git commit -m "Deployed to GitHub Pages"
    git push --force --quiet "https://${GH_TOKEN}@${GH_REF}" gh-pages > /dev/null 2>&1
    echo "-- Deployment done --"
    )
else
    echo "-- will only push docs from master --"
fi
