#!/usr/bin/env bash
#
# A very simple script to run the gunicorn 
# web server for testing.
#

# Load our environment.
export APP_VENV=app-venv
if [ ! -d "${APP_VENV}" ]; then
  echo "Didn't find the python virtual environment (${APP_VENV}). Run ./mkvenv.sh"
  exit 1
else
  source ${APP_VENV}/bin/activate
fi

# Run it!
gunicorn --config gunicorn.conf.py web:app

#
# EOF
