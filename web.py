#!/usr/bin/env python3
#
# ERPC HP/EV Transformer Overloading Evaluation Tool Web Application
#
import os, uuid
from werkzeug.utils import secure_filename
from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, flash, request, render_template, session
from flask_session import Session
from cachelib.file import FileSystemCache


# Start the web application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('WA_SECRET_KEY', default='THISISNOTSAFE238493418')
app.config['UPLOAD_FOLDER'] = os.getenv('WA_OUTPUT_DIR', default='output')
ALLOWED_EXTENSIONS = {'xlsx'}

# Configure session tracking
SESSION_PERMANENT = False
SESSION_USE_SIGNER = True
SESSION_TYPE = 'cachelib'
SESSION_SERIALIZATION_FORMAT = 'json'
SESSION_FILE_DIR = f"{app.config['UPLOAD_FOLDER']}/sessions"
SESSION_CACHELIB = FileSystemCache(threshold=512, cache_dir=SESSION_FILE_DIR), app.config.from_object(__name__)
Session(app)


# Get session uuid (or create on if not set).
def get_session_uuid():
    if not 'user_uuid' in session:
        session['user_uuid'] = uuid.uuid4()
    return session['user_uuid']


# Check uploaded file extension matches allowed set.
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def page_landing():
    user_session_uuid = get_session_uuid()
    return render_template("landing.html")


@app.route("/form", methods = ['GET', 'POST'])
def page_form():
    user_session_uuid = get_session_uuid()
    form_fallback     = False

    # Handle POST case first. We'll fall back to input form
    # if there's something wrong with the user supplied data.
    if request.method == 'POST':
        #tfot = TFOverload_Tool(args.amidata, args.tcinfo, args.evpen, args.hppen, args.output)
        #return render_template("calc.html")
        return "Not implemented"

    # Either show form or fallback to form input because of a problem.
    if request.method == 'GET' or form_fallback:
        return render_template("form.html")


# Deal with robots
@app.route("/robots.txt")
def robots_txt():
    return render_template("robots.txt")


#
# EOF
