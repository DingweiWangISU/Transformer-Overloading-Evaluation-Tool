#!/usr/bin/env bash
#
# For local development and testing, this script will create 
# a python virtual environment that can be used.
#

export APP_VENV=app-venv
python3 -m venv ${APP_VENV}
source ${APP_VENV}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Run source ${APP_VENV}/bin/activate to use the virtual environment."

#
# EOF
