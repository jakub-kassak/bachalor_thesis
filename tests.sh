#!/bin/sh

echo '### mypy results ###'
mypy --python-version=3.11 --ignore-missing-imports /app/src
printf "\n### tests results ###\n"
cd /app/tests || exit 1
python3.11 -m unittest discover -v .