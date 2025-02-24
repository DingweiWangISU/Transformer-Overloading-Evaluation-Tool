#!/usr/bin/env python3
#
# ERPC HP/EV Transformer Overloading Evaluation Tool Web Application
#
# Useful environment variables:
#  - WA_SECRET_KEY
#  - WA_OUTPUT_DIR
#  - WA_ISU_BRANDING
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

# Configure server-side session tracking
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_TYPE'] = 'cachelib'
app.config['SESSION_SERIALIZATION_FORMAT'] = 'json'
app.config['SESSION_FILE_DIR'] = f"{app.config['UPLOAD_FOLDER']}/sessions"
app.config['SESSION_CACHELIB'] = FileSystemCache(threshold=512, cache_dir=app.config['SESSION_FILE_DIR'])
Session(app)

# Misc settings
ALLOWED_EXTENSIONS = {'xlsx'}
USE_ISU_BRANDING = True if os.getenv('WA_ISU_BRANDING', default='T') == "T" else False


# Get session uuid (or create on if not set).
def get_session_uuid():
    if not 'user_uuid' in session:
        session['user_uuid'] = uuid.uuid4()
    return session['user_uuid']


# Only require one captcha per session. 
def has_existing_captcha():
    if not 'captcha_done' in session:
        return "N"
    else:
        return "Y"


# Check uploaded file extension matches allowed set.
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# The landing page should describe the purpose of the 
# application and how to generate the input files.
@app.route("/")
def page_landing():
    user_session_uuid = get_session_uuid()
    return render_template("landing.html", USE_ISU_BRANDING=USE_ISU_BRANDING)


# The form page is the heart of the application. It
# allows the user to upload data which is then either
# processed or rejected (returned to the form)
@app.route("/form", methods = ['GET', 'POST'])
def page_form():
    user_session_uuid = get_session_uuid()
    form_fallback     = False

    # Set form defaults.
    userformdata = {
      "penetration_hp": 10,
      "penetration_ev": 20,
      "file_amidata": "",
      "file_tfcinfo": "",
      "captcha_done": has_existing_captcha()
    }

    # Handle POST case first. We'll fall back to input form
    # if there's something wrong with the user supplied data.
    if request.method == 'POST':

        #tfot = TFOverload_Tool(args.amidata, args.tcinfo, args.evpen, args.hppen, args.output)
        # calc should show results, allow result download, and show a truncated form for 
        # redoing the calculation with different options.
        #return render_template("calc.html", USE_ISU_BRANDING=USE_ISU_BRANDING, userformdata=userformdata)
        return "Not implemented"

    # Either show form or fallback to form input because of a problem.
    if request.method == 'GET' or form_fallback:
        print(f"brandit={USE_ISU_BRANDING}")
        return render_template("form.html", USE_ISU_BRANDING=USE_ISU_BRANDING, userformdata=userformdata)


# Deal with robots
@app.route("/robots.txt")
def robots_txt():
    return render_template("robots.txt")


#
# EOF
