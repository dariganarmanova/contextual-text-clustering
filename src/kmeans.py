import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv('../data/preprocessed/bbc_encoded.csv')
ground_truth_labels = df['label_encoded'].values

import os
file_path = input("Enter the path to the .npy file: ").strip()
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file '{file_path}' does not exist.")

word2vec = np.load(file_path)
X = torch.tensor(word2vec, dtype=torch.float32)

def euclidean_distance(x, y):
    return torch.cdist(x, y, p=2)

def kmeans_plus_plus(X, k):
    n_samples = X.shape[0]
    centroids = []
    first_idx = torch.randint(0, n_samples, (1,)).item()
    centroids.append(X[first_idx])
    
    for _ in range(1, k):
        dist_sq = torch.stack([torch.norm(X-c, dim=1) ** 2 for c in centroids], dim=1)
        min_dist = torch.min(dist_sq, dim=1).values
        probs = min_dist / torch.sum(min_dist)
        next_idx = torch.multinomial(probs, 1).item()
        centroids.append(X[next_idx])
    
    return torch.stack(centroids)

def kmeans_single_run(X, k, max_iters=100, tol=1e-4):
    """Run KMeans once and return centroids, labels, and inertia"""
    centroids = kmeans_plus_plus(X, k)
    
    for iteration in range(max_iters):
        distances = euclidean_distance(X, centroids)
        labels = torch.argmin(distances, dim=1)
        
        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                new_centroids.append(X[torch.randint(0, X.shape[0], (1,)).item()])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        
        new_centroids = torch.stack(new_centroids)
        centroid_shift = torch.norm(centroids - new_centroids, dim=1).sum()
        
        centroids = new_centroids
        
        if centroid_shift < tol:
            print(f"Converged in {iteration+1} iterations")
            break
    
    distances = euclidean_distance(X, centroids)
    min_distances = torch.min(distances, dim=1).values
    inertia = torch.sum(min_distances ** 2).item()
    
    return centroids, labels, inertia

def kmeans_best_of_n(X, k, n_runs=5, max_iters=100, tol=1e-4):
    """Run KMeans n times and return the best result based on inertia"""
    best_centroids = None
    best_labels = None
    best_inertia = float('inf')
    
    print(f"Running KMeans {n_runs} times to find the best clustering...")
    
    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...")
        centroids, labels, inertia = kmeans_single_run(X, k, max_iters, tol)
        
        print(f"  Inertia: {inertia:.4f}")
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels
            print(f"  New best run!")
    
    print(f"\nBest run had inertia: {best_inertia:.4f}")
    return best_centroids, best_labels

k = 5
centroids, labels = kmeans_best_of_n(X, k, n_runs=5)

print("Centroids shape:", centroids.shape)
print("Labels shape:", labels.shape)

ari_score = adjusted_rand_score(ground_truth_labels, labels.numpy())
nmi_score = normalized_mutual_info_score(ground_truth_labels, labels.numpy())

print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X.numpy())

plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=15, alpha=0.8)
plt.colorbar(scatter, label="Cluster")


centroids_2d = np.zeros((k, 2))
for i in range(k):
    cluster_points = X_2d[labels.numpy() == i]
    if len(cluster_points) > 0: 
        centroids_2d[i] = np.mean(cluster_points, axis=0)

plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=100, alpha=0.8, marker='X')

plt.title(f"t-SNE visualization of word2vec embeddings clustered with k-means\nARI={ari_score:.4f}, NMI={nmi_score:.4f}")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig('kmeans_clustering.png')
plt.show()

print("\nCluster distributions:")
for i in range(k):
    count = torch.sum(labels == i).item()
    percentage = count / len(labels) * 100
    print(f"Cluster {i}: {count} samples ({percentage:.1f}%)")

print("\nGround truth label distribution:")
unique_labels, counts = np.unique(ground_truth_labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    percentage = count / len(ground_truth_labels) * 100
    print(f"Label {label}: {count} samples ({percentage:.1f}%)")

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=15, alpha=0.8)
plt.title(f"KMeans Clustering\nk={k}, ARI={ari_score:.4f}")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.subplot(1, 2, 2)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ground_truth_labels, cmap='tab10', s=15, alpha=0.8)
plt.title("Ground Truth Labels")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.tight_layout()
plt.savefig('clustering_comparison.png')
plt.show()