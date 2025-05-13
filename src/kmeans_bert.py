import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the data
contextual_vectors = np.load('../output/contextual_vectors.npy')
X = torch.tensor(contextual_vectors, dtype=torch.float32)  # shape: [2127, 768]

def euclidean_distance(x, y):
    # x: [n, d], y: [k, d]
    # returns [n, k] distance matrix
    return torch.cdist(x, y, p=2)

def kmeans_plus_plus_init(X, k):
    n_samples = X.shape[0] #2172
    centroids = []

    # Randomly select the first centroid
    first_idx = torch.randint(0, n_samples, (1,)).item()
    centroids.append(X[first_idx])

    for _ in range(1, k):
        # Compute distances from each point to the nearest centroid
        dist_sq = torch.stack([torch.norm(X - c, dim=1) ** 2 for c in centroids], dim=1)
        min_dist_sq = torch.min(dist_sq, dim=1).values

        # Probability distribution proportional to distance squared
        probs = min_dist_sq / torch.sum(min_dist_sq)

        # Sample next centroid index
        next_idx = torch.multinomial(probs, 1).item()
        centroids.append(X[next_idx])


    return torch.stack(centroids)

def kmeans(X, k, max_iters=100, tol=1e-4):
    # Step 1: Initialize centroids using kmeans++
    centroids = kmeans_plus_plus_init(X, k)

    for iteration in range(max_iters):
        # Step 2: Assign each point to the nearest centroid
        distances = euclidean_distance(X, centroids)  # [n, k]
        labels = torch.argmin(distances, dim=1)       # [n]

        # Step 3: Compute new centroids
        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                # Rare case: re-initialize to a random point
                new_centroids.append(X[torch.randint(0, X.shape[0], (1,)).item()])
            else:
                new_centroids.append(cluster_points.mean(dim=0))
        new_centroids = torch.stack(new_centroids)

        # Step 4: Check for convergence
        centroid_shift = torch.norm(centroids - new_centroids, dim=1).sum()
        if centroid_shift < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break

        centroids = new_centroids

    return centroids, labels

# Run KMeans
k = 5
centroids, labels = kmeans(X, k)

print("Centroids shape:", centroids.shape)  # Should be [5, 768]
print("Labels shape:", labels.shape)        # Should be [2127]

print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X.numpy())

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels.numpy(), cmap='tab10', s=10)
plt.colorbar(scatter)
plt.title("t-SNE Visualization of BERT Embeddings Clustered with KMeans")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.tight_layout()
plt.show()