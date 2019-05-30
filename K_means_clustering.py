import NLP_ch2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# global vars
k = 4
random_state = 42
categories = ['talk.religion.misc', 'comp.graphics', 'sci.space', 'alt.atheism']

def k_means(k, random_state):
	groups = NLP_ch2.fetch_20newsgroups(categories=categories)
	data_tfidf, vector_tfidf = NLP_ch2.process_docs(groups, None)
	print('Kmeans fitting')
	kmeans = KMeans(n_clusters=k, random_state=random_state)
	kmeans.fit(data_tfidf)
	print('Kmeans complete\n')
	clusters = kmeans.labels_
	centroids = kmeans.cluster_centers_
	return clusters, groups, centroids, vector_tfidf


def examine_clusters(k, clusters, groups, centroids, vector_tfidf):
	cluster_counts = Counter(clusters)
	labels = groups.target_names
	terms = vector_tfidf.get_feature_names()
	for i in range(k):
		print('cluster_{}: {}'.format(i, cluster_counts[i]))
		cluster_groups = Counter(groups.target[np.where(clusters == i)])
		for target, samples in cluster_groups.items():
			print('{} : {} samples'.format(labels[target], samples))
		print('Top 10 terms')
		print(centroids[0])
		for term in centroids[i].argsort()[-10:]:
			print(terms[term], end=' ')
		print()


def main():
	clusters, groups, centroids, vector_tfidf = k_means(k, random_state)
	examine_clusters(k, clusters, groups, centroids, vector_tfidf)

main()