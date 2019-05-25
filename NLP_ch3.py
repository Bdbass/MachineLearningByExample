import numpy as np
import math
from sklearn import datasets
from matplotlib import pyplot as plt

#get data and simplify it 
def fetch_data(all_data):
	iris = datasets.load_iris()
	if all_data:
		data = iris.data
	else: 
		data = iris.data[:, 2:4]
	target_groups = iris.target
	return data, target_groups

#graph data
def plot_original_data(k, data):
	target_groups = data[1]
	data_2d = data[0]
	for j in range(k):
		target = [i for i, x in enumerate(target_groups) if x == j]
		plt.scatter(data_2d[target, 0], data_2d[target, 1])


def intialize_centriods(k, data):
	r_indexes = np.random.choice(len(data), k)
	return data[r_indexes]


def visualize_centroids(data, centroids):
	centroids = np.array(centroids)
	plt.scatter(data[:,0], data[:, 1])
	plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s = 200)
	plt.show()


def euclid_dist(a, b):
	dist_sum = 0
	for i in range(len(a)):
		dist_sum += pow(a[i] - b[i], 2)
	return math.sqrt(dist_sum)


def closest_centroid(a, centroids):
	closest = 0
	min_dist = euclid_dist(a, centroids[0])

	for i, centroid in enumerate(centroids):
		if i == 0:
			continue
		dist = euclid_dist(a, centroid)
		if dist < min_dist:
			min_dist = dist
			closest = i

	return closest


def upate_cluster(data, centroids):
	bins = [closest_centroid(x,centroids) for x in data]
	new_centroids = []
	for i in range(len(centroids)):
		data_slice = [data[j] for j,x in enumerate(bins) if x == i]
		new_centroids.append(np.mean(data_slice, axis=0))
	return new_centroids, bins


def k_mean_clustering(tol, max_iter, data, k):
	i = 0
	temp_tol = 1
	centroids = intialize_centriods(k, data)
	print(centroids)
	while i < max_iter and temp_tol > tol:
		temp_centroids, bins = upate_cluster(data, centroids)
		temp_tol = np.linalg.norm(np.array(centroids) - np.array(temp_centroids))
		i += 1
		centroids = temp_centroids
		print('iterations:', i)
		print('tolerance: ', temp_tol)
		print('centroids: ', centroids)
		#visualize_centroids(data, centroids)
	return centroids, bins


def centroids_and_original_clusters(k, data, centroids):
	plot_original_data(k, data)
	centroids = np.array(centroids)
	plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s = 200)
	plt.show()

def sse(bins, data, centroids):
	squared_error = 0
	for i, x in enumerate(centroids):
		temp_bin = [data[j] for j, y in enumerate(bins) if y == i]
		squared_error += np.linalg.norm(np.array(temp_bin)-np.array(x))
	return squared_error

def visualize_elbow(k_list, sse_array):
	plt.plot(k_list, sse_array)
	plt.show()


def elbow_method(k_start, k_end):
	data = fetch_data(True)
	sse_array = []
	for i in range(k_start, k_end):
		centroids, bins = k_mean_clustering(0.0001, 100, data[0], i)
		sse_array.append(sse(bins, data[0], centroids))
	visualize_elbow(np.arange(k_start, k_end), np.array(sse_array))

def main():
	# visualize clustering for 2D
	# data = fetch_data(False)
	# centroids = k_mean_clustering(0.0001, 100, data[0], 3)[0]
	# centroids_and_original_clusters(3, data, centroids)

	elbow_method(1,6)
	

main()




