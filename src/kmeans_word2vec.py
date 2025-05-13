import torch 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv('../data/preprocessed/bbc_encoded.csv')
ground_truth_labels = df['label_encoded'].values
word2vec = np.load('../output/sentence_vectors.npy')
X = torch.tensor(word2vec, dtype=torch.float32)

def euclidean_distance(x,y):
    return torch.cdist(x,y,p=2)

def kmeans_plus_plus(X,k):
    n_samples = X.shape[0]
    centroids = []
    first_idx = torch.randint(0,n_samples,(1,)).item()
    centroids.append(X[first_idx])
    for _ in range(1,k):
        dist_sq = torch.stack([torch.norm(X-c, dim=1) ** 2 for c in centroids], dim=1)
        min_dist = torch.min(dist_sq, dim=1).values
        probs = min_dist/torch.sum(min_dist)
        next_idx = torch.multinomial(probs,1).item()
        centroids.append(X[next_idx])
    return torch.stack(centroids)

def kmeans(X,k,max_iters=100, tol=1e-4):
    centroids = kmeans_plus_plus(X,k)
    for iteration in range(max_iters):
        distances = euclidean_distance(X,centroids)
        labels = torch.argmin(distances,dim=1)
        new_centroids = []
        for i in range(k):
            cluster_points = X[labels==i]
            if len(cluster_points) == 0:
                new_centroids.append(X[torch.randint(0,X.shape[0],(1,)).item()])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        new_centroids = torch.stack(new_centroids)
        centroid_shift = torch.norm(centroids-new_centroids, dim=1).sum()
        if centroid_shift < tol:
            print(f"Converged in {iteration+1} iterations")
            break 
        centroids = new_centroids
    return centroids,labels

k = 5 
centroids, labels = kmeans(X,k)
print("Centroids shape:", centroids.shape)
print("Labels shape:", labels.shape)
ari_score = adjusted_rand_score(ground_truth_labels, labels.numpy())
nmi_score = normalized_mutual_info_score(ground_truth_labels, labels.numpy())

print(f"Adjusted Rand Index (ARI): {ari_score}")
print(f"Normalized Mutual Information (NMI): {nmi_score}")
print("Running t-SNE...")
tsne = TSNE(n_components=2,random_state=42,perplexity=30)
X_2d = tsne.fit_transform(X.numpy())

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_2d[:,0],X_2d[:,1], c=labels.numpy(), cmap='tab10', s=10)
plt.colorbar(scatter)
plt.title("t-sne visualization of word2vec embeddings clustered with k-means")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.show()
