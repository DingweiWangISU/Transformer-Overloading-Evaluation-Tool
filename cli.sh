#!/usr/bin/env bash
#
# A very simple script to run the CLI.
#

# Load our environment.
source common.sh
app_venv_check
source ${APP_VENV}/bin/activate

# Run it!
python3 cli.py $@

#
# EOF
