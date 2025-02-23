#!/usr/bin/env python3
#
# 
#
from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def landing_page():
    return render_template("landing.html")
    #return "inspector"

# 
#tfot = TFOverload_Tool()
#return tfot.whoami()

#
# EOF
