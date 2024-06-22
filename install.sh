#!/bin/bash
python3 -m virtualenv .venv
source .venv/bin/activate
pip install cython==3.0.10
pip install -r requirements.txt
pip install huggingface-hub==0.23.3