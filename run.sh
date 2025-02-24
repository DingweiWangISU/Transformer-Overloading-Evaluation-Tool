#!/usr/bin/env bash
#
# A very simple script to run the gunicorn 
# web server for testing.
#

# Load our environment.
source common.sh
app_venv_check
source ${APP_VENV}/bin/activate

# Run it!
echo "Running gunicorn for web app."
echo "Control-C to exit when done."
echo ""
# web:app means run Flask application called "app" in "web.py".
gunicorn --config gunicorn.conf.py web:app

#
# EOF
