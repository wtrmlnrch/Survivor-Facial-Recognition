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
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__)) # the path to this directory
# AI3_FALL2024
# FloridaSouthernCS
DATADIR = os.path.join(os.path.dirname(os.path.dirname(ROOT)), 'FloridaSouthernCS', 'csc4510-f24-hw1-schwartzinators', 'data')
Prof_DataDir = os.path.join(os.path.dirname(DATADIR), 'data', 'professors')
Surv_DataDir = os.path.join(os.path.dirname(DATADIR), 'data','survivor')
HEIGHT = 70
WIDTH = 70
THRESHOLD = 0.9
NUM_SEASONS = 46

parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')
parser.add_argument('--threshold', type=int, help='Set value for variance thrshold for PCA')
parser.add_argument('--seasons', type=int, help='Set the number of suvivor seasons used in the dataset')


def main(args):
     
	if args.debug:
		pdb.set_trace()

	if args.threshold:
		THRESHOLD = args.threshold
	
	if args.seasons:
		NUM_SEASONS = args.seasons
          
	#Apply **PCA** to the _Survivor_ faces dataset in order to reduce dimensionality 
	# while maintaining at least 90% of the original variance. You are encouraged to use the
	#  [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) methods 
	# in the scikit-learn library.
     
	# load data from a directory 
	prof_data, prof_labels, prof_names= load(Prof_DataDir)
	surv_data, surv_labels, season_labels = load(Surv_DataDir)
	season_labels = np.char.replace(season_labels, 'S', '')
	season_labels = season_labels.astype(int)
	
	# Determine the mean face of the data set
	mean_face = np.mean(surv_data, axis=1)

	# Set the PCA to hold a 90% variance threshold
	pca = PCA(THRESHOLD)

	# Fit the made PCA to our survivor data transposed
	pca.fit(surv_data.T)

	# Utilize the new pca and the averagesurvivor face to ccollect the average suvivor faced in the PCA face space
	mean_face_pca = pca.inverse_transform(pca.transform([mean_face]))
	mean_image_reshaped_PCA = mean_face_pca.reshape((HEIGHT,WIDTH))

	# Visualize mean face
	plt.figure(figsize=(8,4))
	plt.subplot(1,1,1)
	plt.imshow(mean_image_reshaped_PCA, cmap='gray')
	plt.show()
     
	# 2. Which professor looks least like a face according to the underlying facial features in the 
	# Survivor dataset?_ To answer this question, reconstruct each professor's face using the limited 
	# number of principal components from (1), then compute the Euclidean distance from the
    # reconstructed face to the original. Largest distance indicates least likely to be a "face".
          
	# Reconstruct each professor's face into the face space
	reconstructed_faces = pca.inverse_transform(pca.transform(prof_data.T)) 
     
	# Visualize each professor now in their reduced form
	montage_profs = reconstructed_faces.reshape(len(prof_labels),HEIGHT,WIDTH)
	prof_montage = montage(montage_profs, grid_shape=(1,len(prof_labels)))
	plt.imshow(prof_montage, cmap='gray')
	plt.axis('off')
	plt.title('Roberson | Ngo | Cazalas | Burke | Eicholtz')
	plt.show()
          
	# Calculate the distance of the professors in their reduced form from their original form
	distances = np.linalg.norm(prof_data.T - reconstructed_faces, axis=1)

	# Find the professor with the largest distance from its reduced self
	least_face_index = np.argmax(distances)
	least_face_distance = distances[least_face_index]

	# Print out the result
	print(f"Professor {prof_labels[least_face_index]} looks least like a survivor contestant. Distance: {least_face_distance}")

	# 3. _Which professor is most likely to be the next host of Survivor?_ 
	# To answer this question, project each professor into the reduced "Survivor face space"
	# and apply **nearest neighbor** classification to see who looks most similar to Jeff Probst.

    # Project each professor into the reduced "Survivor face space"
	prof_data_pca = pca.transform(prof_data.T)

	# Find the jeff probst image data and project him into the survivor face space
	jeff_idx = np.where(surv_labels == 'Probst')[0][0]  # Replace with actual label
	jeff_pca = pca.transform(surv_data[:, jeff_idx].reshape(1, -1))

	# Use Nearest Neighbors and fit it to the professor specific pca
	nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
	nn.fit(prof_data_pca)

	# Determine the distance of each professor in the face space from jeff probst
	distances, idc = nn.kneighbors(jeff_pca)

	# Find and print the most similar professor
	jeff_professor_idx = idc[0][0]
	print(f"The professor with the highest chance of hosting survivor is: {prof_labels[jeff_professor_idx]}")
	print(f"Distance to Jeff: {distances[0][0]}")
     

	#4. _Which season would each professor most likely be on?_ To answer this question, use 
	# **k-means clustering** on the PCA-reduced _Survivor_ faces, then assign each of the PCA-reduced 
	# professor faces to the nearest cluster. The average season of _Survivor_ castaways in the cluster
	#  (not including Jeff Probst) is the assigned season for that professor.
    
	# Fit the pca to survivor data
	surv_pca = pca.fit_transform(surv_data.T).T 

	# Use sume squared error to determine the best number of clusters for the dataset
	sse = [] #SUM OF SQUARED ERROR
	for k in range(1, NUM_SEASONS):
		km = KMeans(n_clusters=k, random_state=2)
		km.fit(surv_pca.T)
		sse.append(km.inertia_)
      
	# Visualize each number of K along with its associated SSE
	sns.set_style("whitegrid")
	g=sns.lineplot(x=range(1,46), y=sse)

	g.set(xlabel ="Number of K clusters", 
		ylabel = "Sum Squared Error", 
		title ='Elbow Method')
	plt.show()

	# Get the slope of each k clusters to determine which number of k is most linear
	slopes = np.gradient(sse)
	dis_to_one = np.subtract(slopes, -1)
	most_linear = np.max(dis_to_one)
	n_clusters = np.where(dis_to_one == most_linear)[0].tolist()[0]


	# Using found optimal number of clusters apply k-means clustering to survivor pca
	kmeans = KMeans(n_clusters=n_clusters, random_state=0)
	kmeans.fit(surv_pca.T) 

	# Get assigned cluster labels for each image
	cluster_labels = kmeans.labels_
     
	# Visualize the cluster
	plt.figure(figsize=(10, 7))
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
    
	# Get cluster assignments
	cluster_assignments = kmeans.labels_

	# Initialize list to store average labels for each cluster
	common_labels = []
      
	for clusters in prof_clusters:
		# Find indices of data points in the current cluster
		indices = np.where(cluster_assignments == clusters)[0]
		
		# Extract season labels of data points in the current cluster
		cluster_labels = []
		for idx in indices:
			cluster_labels.append(season_labels[idx])

		# Calculate the most common season label
		common_label = stats.mode(cluster_labels, keepdims=False)[0]
		common_labels.append(common_label)

    # Print professor assignments to seasons/clusters
	print("Professor assignments to clusters/seasons:")
	for i, label in enumerate(prof_clusters):
		print(f"Professor {prof_labels[i]} is closest to Season {common_labels[i]}.")
            

	# 5. Which professor is most likely to win Survivor? Be creative! You must justify your answer 
	# to this question in a quantitative way using the results of PCA on the Survivor dataset.

	# all the winners of survivor
	winners = ['Hatch', 'Wesson', 'Zohn', 'Towery', "Heidik", "Morasca", 
	"Diaz-Twine", "Brkich", "Daugherty", "Westman", "Boatwright", 
	"Baskauskas", "Kwon", "Cole", "Herzog", "Shallow", "Crowley", 
	"Thomas", "White", "Diaz-Twine", "Birza", "Mariano", "Clarke", 
	"Spradlin", "Stapley", "Cochran", "Apostol", "Vlachos", 'Anderson', 
	"Holloway", "Collins", "Fitzgerald", "Klein", "Lacina", "Driebergen",
	"Holland", 'Wilson', 'Underwood', 'Sheehan', "Vlachos",
	"Casupanan", "Oketch", "Gabler", "Arocho", "Valladares", "Petty"]

	# Find winners indexes
	winner_idxs = []

	# Aprraoch Drawbacks:
	# doesn't check for possible wrong winners
	# however removes duplicate indexes of winners
	for name in winners:
		idxs = np.where(surv_labels == name)
		for ids in idxs[0]:
			winner_idxs.append(ids)
	winner_idxs = list(set(winner_idxs))
	winner_idxs.sort()

	# creates the winners data and label sets based off of surv_data and surv_labels
	winners_data, winners_labels = [0]*len(winner_idxs), [0]*len(winner_idxs)
	for i in range(len(winner_idxs)):
		winners_data[i], winners_labels[i] = surv_data.T[winner_idxs[i]], surv_labels[winner_idxs[i]]
	winners_data, winners_labels = np.array(winners_data), np.array(winners_labels)

	# use the clusters created by k-means for some fun stuff (:
	# fun stuff: check to see which of the professor's clusters has the most winners
	profs_winners = []
	
	for clusters in prof_clusters:
		# Find indices of data points in the current cluster
		indices = np.where(cluster_assignments == clusters)[0]
		possible_winners = np.intersect1d(indices, winner_idxs)

		# in the case that the possible winner is not an actual winner we delete it 
		# since we only check for last name there is the possibility that there could 
		# be false winners
		for win in possible_winners:
			if surv_labels[win] != winners[season_labels[win]-1]:
				np.delete(possible_winners, np.where(possible_winners == win))
		profs_winners.append(len(np.intersect1d(indices, winner_idxs)))
	
	# gets the possible multiple professors who may be the winner
	profs_winners = np.array(profs_winners)
	winning_profs = np.where(profs_winners == max(profs_winners))
	print(f'The professor(s) who may be most likely to win Survivor: \n{np.array2string(prof_labels[winning_profs], separator=" | ")}')
	
	if len(winning_profs[0]) > 1:
		# find the mean winner face
		mean_winner = np.mean(winners_data.T, axis=1)
		pca = PCA(0.9)
		pca.fit(winners_data)
		mean_winner_pca = pca.inverse_transform(pca.transform([mean_winner]))
		
		#pdb.set_trace()
		mean_winner_reshaped_PCA = mean_winner_pca.reshape((HEIGHT,WIDTH))

		# plotting the image
		plt.rcdefaults()
		plt.figure(figsize=(8,4))
		plt.subplot(1,1,1)
		plt.imshow(mean_winner_reshaped_PCA, cmap='gray')
		plt.show()

		# calculate the euclidean distances between each professors face and the mean winners face
		distances = []
		for prof in prof_data.T:
			distances.append(np.linalg.norm(mean_winner - prof))

		# get the least out of our two already winning professors to find the one closest to the mean winners face
		prof_winner_idx = np.where(distances == min(distances[winning_profs[0][0]], distances[winning_profs[0][1]]))[0][0]
		print(f"The professor who is most likely to win Survivor based on the closest Euclidean distance between their face \nand the mean winner's face is: Dr. {prof_labels[prof_winner_idx]}!")



# Function to load and process the image data
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
    season_labels = np.array([file.split('_')[0].split('.')[0] for file in files])

    return data, labels, season_labels

if __name__ == "__main__":
	main(parser.parse_args())
