#!/usr/bin/env bash
#
# Common definitions for scripts.
#

export APP_VENV=app-venv

function app_venv_check() {
  if [ ! -d "${APP_VENV}" ]; then
    echo "Didn't find the python virtual environment (${APP_VENV}). Run ./mkvenv.sh"
    exit 1
  fi
}

#
# EOF
