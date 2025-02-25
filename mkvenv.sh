#!/usr/bin/env bash
#
# For local development and testing, this script will create 
# a python virtual environment that can be used.
#

# Load our environment.
source common.sh

# If "clean" is passed, delete existing ${APP_VENV}
if [ "$1" == "clean" ]; then
  echo "Cleaning (removing) old python virtual environment (${APP_VENV})."
  rm -rf "${APP_VENV}"
fi

# Only create if there isn't an existing ${APP_VENV}
if [ ! -d "${APP_VENV}" ]; then
  python3 -m venv ${APP_VENV}
  source ${APP_VENV}/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Existing python virtual environment (${APP_VENV}) found. Delete and run ./mkvenv.sh to recreate."
  exit 1
fi

echo "Run [source ${APP_VENV}/bin/activate] to use the virtual environment."

#
# EOF
