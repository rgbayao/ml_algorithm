import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    sides = p1 - p2

    inner_product = np.dot(sides, sides)

    return np.sqrt(inner_product)

class KMeans:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.n_features = 2
        self.data_clusters = np.zeros(0)
        self.cluster_centroids = np.zeros(0)

    def fit(self, data):
        self.data = data
        self.n_features = self.data.shape[1]
        self.data_clusters = np.zeros(len(data))

        self._start_clusters()

        counter = 0
        while counter < self.max_iter:
            self._update_data_clusters()
            self._update_cluster_centroids()
            counter += 1

    def _start_clusters(self):
        feature_min = np.min(self.data, axis=0)
        feature_max = np.max(self.data, axis=0)
        self.cluster_centroids = np.zeros((self.k, self.n_features))
        for i in range(0, self.k):
            self.cluster_centroids[i] = np.random.random(self.n_features) * (feature_max - feature_min) + feature_min

    def _update_data_clusters(self):
        for i in range(0, len(self.data)):
            closest_cluster = self._find_closest_cluster(self.data[i])
            self.data_clusters[i] = closest_cluster

    def _find_closest_cluster(self, data):
        closest_distance = -1
        this_cluster = -1
        for i in range(0, self.k):
            distance = euclidean_distance(data, self.cluster_centroids[i])
            if distance < closest_distance or closest_distance == -1:
                closest_distance = distance
                this_cluster = i
        return this_cluster

    def _update_cluster_centroids(self):
        for i in range(0, self.k):
            data_filter = self.data_clusters == i
            cluster_data = self.data[data_filter]
            self.cluster_centroids[i] = np.mean(cluster_data, axis=0)
            
    def show(self):
        fig = plt.figure()
        for i in range(0,self.k):
            data_lines = self.data_clusters == i
            plt.scatter(self.data[data_lines][:,0], self.data[data_lines][:,1], label = f'Cluster {i}')
            plt.scatter(self.cluster_centroids[i,0],self.cluster_centroids[i,1], marker = 'o', facecolors = 'none', edgecolors='r')
            plt.legend()
            
    def predict(self, X):
        X_clusters = np.zeros(len(X))
        for i in range(0,len(X)):
            X_clusters[i] = self._find_closest_cluster(X[i])
        return X_clusters
