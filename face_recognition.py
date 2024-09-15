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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
	pca = PCA(0.90) 
	surve_proj = pca.fit_transform(surv_data)
	print(pca)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(surv_data)
     
	plt.scatter(surve_proj[:,0], surve_proj[:,1],c=surv_labels) 
	plt.show()
     
	plt.figure(figsize=(8,8)) 
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) # plot the faces, each image is 64 by 64 pixels 
	for i in range(len(surv_labels)): 
		ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
		ax.imshow(np.reshape(pca.components_[i,:], (64,64)), cmap=plt.cm.bone, interpolation='nearest') 
      
	# for prof in prof_labels:
	# 	reconstructed_images = []
	# 	this_prof = np.where(prof_labels == prof)[0][0]

	# 	# Use only the first `num_weights` components
	# 	weights_subset = weights[:num_eigfaces]
	# 	eigfaces_subset = top_eigfaces[:, :num_eigfaces]

	# 	# Reconstruct the image
	# 	reconstructed_face = np.dot(eigfaces_subset, weights_subset) + mu

	# 	# Reshape and add to the list of images
	# 	reconstructed_face_image = reconstructed_face.reshape(HEIGHT, WIDTH)
	# 	reconstructed_images.append(reconstructed_face_image)

	# 	# Plotting the montage
	# 	fig, axes = plt.subplots(1, len(num_eigfaces), figsize=(15, 5))

	# 	for ax, num_weights, img in zip(axes, num_eigfaces, reconstructed_images):
	# 		ax.imshow(img, cmap='gray')
	# 		ax.set_title(f'{num_weights} PCs')
	# 		ax.axis('off')

	# plt.show()


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
