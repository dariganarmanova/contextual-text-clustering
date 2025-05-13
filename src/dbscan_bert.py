import torch 
import numpy as np 
import matplotlib.plt as plt 
from sklearn.manifold import TSNE

contextual_vectors = np.load('../output/contextual_vectors.npy')
X = torch.tensor(contextual_vectors, dtype=torch.float32)

def euclidean_distance(x,y):
    return torch.cdist(x,y,p=2)

def dbscan(X,eps,min_samples):
    n_samples = X.shape[0]
    labels=torch.full((n_samples,), -1,dtype=torch.int32)
    cluster_id = 0
    distances = euclidean_distance(X,X)
    def region_query(point_idx):
        return torch.where(distances[point_idx] <=eps)[0]
    def expand_cluster(point_idx,neighbors, cluster_id):
        labels[point_idx] = cluster_id
        to_visit = neighbors.tolist()
        while to_visit:
            current_point = to_visit.pop()
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            current_neighbors = region_query(current_point)
            if len(current_neighbors) >= min_samples:
                for neighbor in current_neighbors: 
                    if labels[neighbor] == -1:
                        to_visit.append(neighbor)
                        
