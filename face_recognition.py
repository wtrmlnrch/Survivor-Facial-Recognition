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
from skimage.util import montage
import matplotlib.pyplot as plt
import sys
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns

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
	
	mean_face = np.mean(surv_data, axis=1)
	pca = PCA(0.9)
	surve_proj = pca.fit(surv_data.T)
	mean_face_pca = pca.inverse_transform(pca.transform([mean_face]))
     
	#pdb.set_trace()
	mean_image_reshaped_PCA = mean_face_pca.reshape((70,70))

	plt.figure(figsize=(8,4))
	plt.subplot(1,1,1)
	plt.imshow(mean_image_reshaped_PCA, cmap='gray')
	plt.show()
     
	# 2. Which professor looks least like a face according to the underlying facial features in the 
	# Survivor dataset?_ To answer this question, reconstruct each professor's face using the limited 
	# number of principal components from (1), then compute the Euclidean distance from the
    # reconstructed face to the original. Largest distance indicates least likely to be a "face".
          
	# Reconstruct each professor's face
	reconstructed_faces = pca.inverse_transform(pca.transform(prof_data.T)) #.reshape(5,70,70)
	#pdb.set_trace()
     
	# montage code
	montage_profs = reconstructed_faces.reshape(5,70,70)
	prof_montage = montage(montage_profs, grid_shape=(1,5))
	plt.imshow(prof_montage, cmap='gray')
	plt.axis('off')
	plt.title('Roberson | Ngo | Cazalas | Burke | Eicholtz')
	plt.show()
	#pdb.set_trace()
          
	distances = np.linalg.norm(prof_data.T - reconstructed_faces, axis=1)

	# Find the professor with the largest distance
	least_face_index = np.argmax(distances)
	least_face_distance = distances[least_face_index]

	print(f"Professor {prof_labels[least_face_index]} looks least like a survivor contestant. Distance: {least_face_distance}")

	# 3. _Which professor is most likely to be the next host of Survivor?_ 
	# To answer this question, project each professor into the reduced "Survivor face space"
	# and apply **nearest neighbor** classification to see who looks most similar to Jeff Probst.
     
	#Make NN only used with on Jeff Probst vector and Professor images
    # Project each professor into the reduced "Survivor face space"
	prof_data_pca = pca.transform(prof_data.T)

	# Assuming we know Jeff Probst's index, find his PCA vector
	jeff_idx = np.where(surv_labels == 'Probst')[0][0]  # Replace with actual label
	jeff_pca = pca.transform(surv_data[:, jeff_idx].reshape(1, -1))

	# Apply Nearest Neighbor classification
	nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
	nn.fit(prof_data_pca)
	distances, spots = nn.kneighbors(jeff_pca)

	# Find and print the most similar professor
	jeff_professor_idx = spots[0][0]
	print(spots)
	print(f"The professor with the highest chance of hosting survivor is: {prof_labels[jeff_professor_idx]}")
	print(f"Distance to Jeff: {distances[0][0]}")
     

	#4. _Which season would each professor most likely be on?_ To answer this question, use 
	# **k-means clustering** on the PCA-reduced _Survivor_ faces, then assign each of the PCA-reduced 
	# professor faces to the nearest cluster. The average season of _Survivor_ castaways in the cluster
	#  (not including Jeff Probst) is the assigned season for that professor.
     
	surv_pca = pca.fit_transform(surv_data.T).T 
	# Number of clusters 
	sse = [] #SUM OF SQUARED ERROR
	for k in range(1,46):
		km = KMeans(n_clusters=k, random_state=2)
		km.fit(surv_pca.T)
		sse.append(km.inertia_)
	print(sse)
      
	  
	sns.set_style("whitegrid")
	g=sns.lineplot(x=range(1,46), y=sse)

	g.set(xlabel ="Number of cluster (k)", 
		ylabel = "Sum Squared Error", 
		title ='Elbow Method')
	plt.show()
	n_clusters = min(sse)

	# Apply k-means clustering
	kmeans = KMeans(n_clusters=n_clusters, random_state=0)
	kmeans.fit(surv_pca.T) 

	# Get cluster labels for each image
	cluster_labels = kmeans.labels_
     
	"""Visualize clustering results."""
	plt.figure(figsize=(10, 7))

	# Scatter plot
	plt.scatter(surv_pca[0, :], surv_pca[1, :], c=cluster_labels, cmap='viridis', alpha=0.5)
	plt.colorbar(label='Cluster Label')
	plt.xlabel('PC 1')
	plt.ylabel('PC 2')
	plt.title('Reduced Data Clustering of survivor seasons')
	plt.show()
	# Project professor data into the PCA space
	prof_pca = pca.transform(prof_data.T)

    # Predict clusters for professors
	prof_clusters = kmeans.predict(prof_data_pca)

    # Print professor assignments to clusters
	print("Professor assignments to clusters/seasons:")
	for i, label in enumerate(prof_clusters):
		print(f"Professor {prof_labels[i]} is closest to {label}")

	# Optionally, visualize the clustering results

	
     
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
