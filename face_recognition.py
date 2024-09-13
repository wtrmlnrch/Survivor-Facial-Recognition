# face_recognition.py
"""Project Description
Name: HW 1 Face Recognition
Authors: Sarte & Sustaita
Professor: Dr. Eicholtz
Class: AI 3 - csc4510
Due Date: 9/25/2024

This project will work to build a PCA with k-means clustering to predict how compeitors wil perform in 
the game survivor based on a face-recogniton model. Who's face is most similar to a survivor host and 
there for may become one? 

Goal:
Face recognition is a classic real-world application for **_unsupervised learning_** techniques. 
In this assignment, you will work in pairs to develop Python code that leverages principal component
analysis (PCA) and k-means clustering along with supervised learning methods such as nearest neighbors
and neural networks to explore an image dataset related to the reality TV show _Survivor_.

"""
# raghr

import argparse
import os
import pdb
from matplotlib.image import imread
import numpy as np
import pdb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.abspath(__file__)) # the path to this directory
# AI3_FALL2024
# FloridaSouthernCS
DATADIR = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'FloridaSouthernCS', 'csc4510-f24-hw1-schwartzinators', 'data')
Prof_DataDir = os.path.join(os.path.dirname(DATADIR), 'data', 'professors')
Surv_DataDir = os.path.join(os.path.dirname(DATADIR), 'data','survivor')
HEIGHT = 70
WIDTH = 70

parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')

def main(args):
     
	if args.debug:
		pdb.set_trace()
          
	#Apply **PCA** to the _Survivor_ faces dataset in order to reduce dimensionality 
	# while maintaining at least 90% of the original variance. You are encouraged to use the
	#  [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) methods 
	# in the scikit-learn library.
     
	# load data from a directory 
	prof_data, prof_labels = load(Prof_DataDir)
	surv_data, surv_labels = load(Surv_DataDir)
     
	# starting pca (:
	print(prof_labels)
	print(surv_labels)
     
	# Compute the average survivor
	mu = np.mean(surv_data, axis=1)

    # Reshape the average survivor to 2D
	mu_image = mu.reshape(HEIGHT, WIDTH)

    # Visualize the average survivor face
	plt.imshow(mu_image, cmap='gray')
	plt.title('Average Survivor')
	plt.colorbar()
	plt.show()
     
	# Compute the difference between images and the mean
	A = surv_data - mu[:, np.newaxis]

	# Compute the covariance matrix
	S = np.dot(A.T, A) / A.shape[1]

	# Solve the eigenvalue problem 
	eigvalues, eigfaces = np.linalg.eig(S)  # eigdonuts: (n_samples, n_samples)

	# Sort eigenvalues and corresponding eigenvectors
	sorted_indices = np.argsort(eigvalues)[::-1]
	eigvalues = eigvalues[sorted_indices]
	eigfaces = eigfaces[:, sorted_indices]

	# Compute eigenvectors in the original feature space
	top_eigfaces = np.dot(A, eigfaces)  # Transform eigenvectors

	# Determine how many eigenvectors to keep 
	total_variance = np.sum(eigvalues)
	cumulative_variance = np.cumsum(eigvalues) / total_variance
	variance_threshold = 0.80  # 90% variance
	num_eigfaces = np.searchsorted(cumulative_variance, variance_threshold) + 1

	# Select the top eigenvectors
	top_eigfaces = top_eigfaces[:, :num_eigfaces]  # Shape: (n_features, num_eigendonuts)

	# Project data samples into "donut space"
	projected_data = np.dot(top_eigfaces.T, A)  # Shape: (num_eigendonuts, n_samples)

	print(f"Number of eigendonuts to keep: {num_eigfaces}")
	print("Projected data shape:", projected_data.shape)



def load(directory=DATADIR):
    '''Load data (and labels) from directory.'''
    files = os.listdir(directory)  # extract filenames
    files = [f for f in files if os.path.isfile(os.path.join(directory, f))]  # Ensure we only process files
    n = len(files)  # number of files

    if n == 0:
        print("No files found in the directory.")
        return None, None

    # Load all images into 2D array
    pixels = HEIGHT * WIDTH
    data = np.zeros((pixels, n), dtype='float32')
    for i in range(n):
        filepath = os.path.join(directory, files[i])
        print(f'Loading image: {files[i]}')
        try:
            rgb = imread(filepath)
            
            # NOTE: I added this because some images had a 4th layer (alpha channel perhaps?)
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]

            img = rgb2gray(rgb)
            if img.shape[0] != HEIGHT or img.shape[1] != WIDTH:
                print(f"WARNING!!! Image ({img.shape[0]}, {img.shape[1]}) does not have expected dimensions ({HEIGHT, WIDTH})")
                continue  # Skip this image

            data[:, i] = img.reshape(-1,)
        except Exception as e:
            print(f"Error loading image {files[i]}: {e}")
            continue

    # Extract labels from filenames
    labels = np.array([file.split('_')[-1].split('.')[0] for file in files])

    return data, labels

if __name__ == "__main__":
	main(parser.parse_args())
