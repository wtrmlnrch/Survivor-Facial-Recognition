# face_recognition.py
"""Project Description

This project will work to build a PCA with k-means clustering to predict how compeitors wil perform in 
the game survivor based on a face-recogniton model. Who's face is most similar to a survivor host and 
there for may become one? 

Goal:
Face recognition is a classic real-world application for **_unsupervised learning_** techniques. 
In this assignment, you will work in pairs to develop Python code that leverages principal component
analysis (PCA) and k-means clustering along with supervised learning methods such as nearest neighbors
and neural networks to explore an image dataset related to the reality TV show _Survivor_.

"""
# Add some comments here about what the code does.
# raghr

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
