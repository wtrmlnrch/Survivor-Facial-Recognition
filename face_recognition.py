# face_recognition.py
# Add some comments here about what the code does.

import argparse
import os
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__)) # path to source directory of this file

parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')

def main(args):
	if args.debug:
		pdb.set_trace()

if __name__ == "__main__":
	main(parser.parse_args())
