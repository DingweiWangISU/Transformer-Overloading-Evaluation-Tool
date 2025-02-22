#!/usr/bin/env python3
#
# The cli.py file provides a method to invoke the
# EPRC Transformer Overloading Evaluation Tool 
# without loading the full web application.
#
# This is useful for debugging the underlying 
# application code without dealing with a full 
# web service framework.
#
# It's also beneficial for anyone trying to use
# this tool in an automated fashion.
#

from eprc.tfoverload_tool import TFOverload_Tool

def cli_main():
    tfot = TFOverload_Tool()
    tfot.whoami()

if __name__ == "__main__":
    cli_main()
#
# EOF
