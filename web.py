from eprc.tfoverload_tool import TFOverload_Tool
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def landing_page():
    tfot = TFOverload_Tool()
    return tfot.whoami()

