#!/usr/bin/env python3
#
# 
#
from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/")
def landing_page():
    return render_template("landing.html")
    #return "inspector"


@app.route("/form", methods = ['GET', 'POST'])
def form_page():
    if request.method == 'GET':
        return render_template("form.html")

    if request.method == 'POST':
        #tfot = TFOverload_Tool()
        #return render_template("calc.html")
        return "Not implemented"


# Deal with robots
@app.route("/robots.txt")
def robots_txt():
    return render_template("robots.txt")


#
# EOF
