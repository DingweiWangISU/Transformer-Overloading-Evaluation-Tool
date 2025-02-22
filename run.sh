#!/usr/bin/env bash
#
# A very simple script to run the gunicorn 
# web server for testing.
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
echo ""
echo "Control-C to exit when done."
# web:app means run Flask application called "app" in "web.py".
gunicorn --config gunicorn.conf.py web:app

#
# EOF
