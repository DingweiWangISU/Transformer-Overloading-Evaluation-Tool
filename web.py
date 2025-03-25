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
from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, flash, request, render_template, session, send_file
from flask_session import Session
from cachelib.file import FileSystemCache
from flask_session_captcha import FlaskSessionCaptcha
from flask_wtf import FlaskForm, CSRFProtect
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import Form, BooleanField, StringField, IntegerField, HiddenField, SubmitField, validators


# Start the web application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('WA_SECRET_KEY', default='THISISNOTSAFE238493418')
app.config['UPLOAD_FOLDER'] = os.getenv('WA_OUTPUT_DIR', default='output')

# Configure server-side session tracking
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_TYPE'] = 'cachelib'
app.config['SESSION_SERIALIZATION_FORMAT'] = 'json'
app.config['SESSION_FILE_DIR'] = os.path.join(app.config['UPLOAD_FOLDER'], 'sessions')
app.config['SESSION_CACHELIB'] = FileSystemCache(threshold=512, cache_dir=app.config['SESSION_FILE_DIR'])
Session(app)

# Configure the Captcha
app.config['CAPTCHA_ENABLE'] = True
app.config['CAPTCHA_LENGTH'] = 5
app.config['CAPTCHA_WIDTH'] = 200
app.config['CAPTCHA_HEIGHT'] = 160
# app.config['CAPTCHA_LOG'] = False # log information to terminal
app.config['CAPTCHA_INCLUDE_ALPHABET'] = False
app.config['CAPTCHA_INCLUDE_NUMERIC'] = True
captcha = FlaskSessionCaptcha(app)

# Misc settings
csrf = CSRFProtect(app)
ALLOWED_EXTENSIONS = {'xlsx'}
USE_ISU_BRANDING = True if os.getenv('WA_ISU_BRANDING', default='T') == "T" else False


# Define the main document form.
class TFOTForm(FlaskForm):
    penetration_ev = IntegerField('Electric Vehicle Penetration Percentage')
    penetration_hp = IntegerField('Heat Pump Penetration Percentage')
    #file_use_existing = BooleanField("Use existing AMI Data and TFC Info", default=False)
    file_amidata = FileField('AMI Data (XLSX)', validators=[FileRequired()])
    file_tfcinfo = FileField('TFC Info (XLSX)', validators=[FileRequired()])
    submit = SubmitField("Calculate")


# Get session uuid (or create on if not set).
def get_session_uuid():
    if not 'user_uuid' in session:
        session['user_uuid'] = str(uuid.uuid4())
    return session['user_uuid']


# Only require one captcha per session. 
def has_existing_captcha():
    return False if not 'captcha_done' in session else True


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
    xlsx_output       = None
    png_output        = None

    # Set form defaults.
    form = TFOTForm()
    userformdata = {
        "penetration_hp": 10,
        "penetration_ev": 20,
        "files_good": False,
        "captcha_done": has_existing_captcha()
    }

    # Validate the captcha, if submitted.
    if request.method == 'POST' and not has_existing_captcha():
        if captcha.validate():
            session['captcha_done'] = True
        else:
            flash("Captcha input incorrect or expired.")
            form_fallback = True

    # Handle POST case first. We'll fall back to input form
    # if there's something wrong with the user supplied data.
    #if not form_fallback and form.validate_on_submit():
    if request.method == 'POST' and not form_fallback:
        # Save user values to userformdata for later rendering.
        userformdata["penetration_ev"] = form.penetration_ev.data
        userformdata["penetration_hp"] = form.penetration_hp.data

        # Handle files - Where to save this user's data (temporarily)
        userfilepath = os.path.join(app.config['UPLOAD_FOLDER'], user_session_uuid)
        os.makedirs(userfilepath, exist_ok=True)

        # Handle files - Save files
        if not form.file_amidata.data or not form.file_tfcinfo.data:
            flash("Both AMI-Data.xlsx and TFC-Info.xlsx files are required.")
            form_fallback = True
        else:
            form.file_amidata.data.save(os.path.join(userfilepath, "amidata.xlsx"))
            form.file_tfcinfo.data.save(os.path.join(userfilepath, "tfcinfo.xlsx"))

        if not form_fallback:
            try:
                tfot = TFOverload_Tool(
                    os.path.join(userfilepath, 'amidata.xlsx'),
                    os.path.join(userfilepath, 'tfcinfo.xlsx'),
                    userformdata["penetration_ev"],
                    userformdata["penetration_hp"],
                    userfilepath
                    )
                xlsx_output, png_output = tfot.run()
            except Exception as e:
                flash(f"Calculation Error: {e}")

        # Always fallback to display the output or the error.
        form_fallback = True

    # Either show form or fallback to form input because of a problem.
    if request.method == 'GET' or form_fallback:
        # Set form defaults and update.
        form.penetration_ev.default = userformdata["penetration_ev"]
        form.penetration_hp.default = userformdata["penetration_hp"]
        form.process()
        # Return the completed form to the user.
        return render_template("form.html", USE_ISU_BRANDING=USE_ISU_BRANDING, userformdata=userformdata, form=form, xlsx_output=xlsx_output, png_output=png_output)
    else:
        return "How did you get here?"


# Return Output
@app.route("/out")
def get_output():
    # Find the user output folder.
    user_session_uuid = get_session_uuid()
    userfilepath = os.path.join(app.config['UPLOAD_FOLDER'], user_session_uuid)
    # Only return output if the user has an output folder.
    if os.path.exists(userfilepath):
      # See if the file requested exists.
      reqdoc = request.args.get('d')
      if reqdoc is not None:
        # Check requested doc is something expected.
        reqfile, reqext = os.path.splitext(reqdoc)
        if reqext in [".xlsx", ".png"] and reqfile.startswith("Transformer_Load_Analysis_Results_pen_level_"):
            # Expected document format.
            # Check if file exists
            reqfspec = f"{userfilepath}/{reqdoc}"
            if os.path.isfile(reqfspec):
              # Should be good to return this to the user.
              return send_file(reqfspec, attachment_filename=reqdoc)
    # If anything is wrong, return an empty string.
    return ""


# Deal with robots
@app.route("/robots.txt")
def robots_txt():
    return render_template("robots.txt")


#
# EOF
