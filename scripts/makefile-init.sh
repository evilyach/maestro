#!/bin/bash

PYTHON_INTERPRETER="python3.8"
VENV_DIR=".env"

if ! $PYTHON_INTERPRETER --version; then
    exit 1
fi

$PYTHON_INTERPRETER -m venv $VENV_DIR

source $VENV_DIR/bin/activate

pip install -e .
python -c "import nltk; nltk.download('stopwords')"

deactivate
