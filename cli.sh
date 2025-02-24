#!/usr/bin/env bash
#
# A very simple script to run the CLI.
#

# Load our environment.
source common.sh
if [ ! -d "${APP_VENV}" ]; then
  echo "Didn't find the python virtual environment (${APP_VENV}). Run ./mkvenv.sh"
  exit 1
else
  source ${APP_VENV}/bin/activate
fi

# Run it!
python3 cli.py $@

#
# EOF
