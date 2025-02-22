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

import argparse
from eprc.tfoverload_tool import TFOverload_Tool

def cli_main(args):
    print("Running EPRC Transformer Overloading Evaluation Tool with...")
    print("  AMI Data = " + args.amidata)
    print("  TC Info  = " + args.tcinfo)
    print("  EV Penetration: " + str(args.evpen) + "%")
    print("  HP Penetration: " + str(args.hppen) + "%")
    print("  Output Folder = " + args.output)

    tfot = TFOverload_Tool(args.amidata, args.tcinfo, args.evpen, args.hppen, args.output)
    try: 
        tfot.run()
    except Exception as e:
        print(f"!! ERROR: {e}")
    print(tfot.whoami())

# Actually do stuff when called directly.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This tool estimates which transformers will overload given some percentage of heat pump and electric vehicle load.')
    parser.add_argument('--amidata', default='data/samples/AMI.xlsx', help='XLSX with a full year of hourly AMI data for all the customers (default: %(default)s)')
    parser.add_argument('--tcinfo', default='data/samples/TC.xlsx', help='XLSX with transformer specifications and transformer-customer connectivity (default: %(default)s)')
    parser.add_argument('--evpen', default=20, type=int, help='Percent Electric Vehicle Penetration (default: %(default)s)')
    parser.add_argument('--hppen', default=10, type=int, help='Percent Heat Pump Penetration (default: %(default)s)')
    parser.add_argument('--output', default='output', help='Folder to store output (default: %(default)s)')
    cli_main(parser.parse_args())

#
# EOF
