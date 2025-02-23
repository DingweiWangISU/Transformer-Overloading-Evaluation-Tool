#!/usr/bin/env python3
#
# 
#
from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, render_template

app = Flask(__name__)
#tfot = TFOverload_Tool()


@app.route("/")
def landing_page():
    return render_template("landing.html")
    #return "inspector"


# Deal with robots
@app.route("/robots.txt")
def robots_txt():
    return render_template("robots.txt")


#
# EOF
