import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os 
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedDBSCANClustering:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.best_params = None
        self.best_score = -1
        self.labels_ = None
        self.valid_indices = None  

    def load_and_preprocess_data(self, npy_path, csv_path):
        self.file_stem = os.path.splitext(os.path.basename(npy_path))[0]
        self.X = np.load(npy_path)
        df = pd.read_csv(csv_path)
        self.ground_truth_original = df['label_encoded'].values

        if self.X.shape[0] != self.ground_truth_original.shape[0]:
            min_samples = min(self.X.shape[0], self.ground_truth_original.shape[0])
            self.X = self.X[:min_samples]
            self.ground_truth_original = self.ground_truth_original[:min_samples]

        self.valid_indices = np.arange(len(self.X))
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            valid_rows = ~(np.isnan(self.X).any(axis=1) | np.isinf(self.X).any(axis=1))            
            self.X = self.X[valid_rows]
            self.ground_truth_original = self.ground_truth_original[valid_rows]
            self.valid_indices = self.valid_indices[valid_rows]

        variances = np.var(self.X, axis=0)
        print(f"Min variance: {np.min(variances):.6f}, Max variance: {np.max(variances):.6f}")
        variance_threshold = max(0.001, np.median(variances) * 0.1)
        print(f"Using variance threshold: {variance_threshold:.6f}")

        variance_selector = VarianceThreshold(threshold=variance_threshold)
        X_variance_filtered = variance_selector.fit_transform(self.X)
        print(f"After variance filtering: {X_variance_filtered.shape}")

        if X_variance_filtered.shape[1] < 2:
            X_variance_filtered = self.X

        self.ground_truth = self.ground_truth_original.copy()
        
        self.X_scaled = self.scaler.fit_transform(X_variance_filtered)
        assert self.X_scaled.shape[0] == self.ground_truth.shape[0], \
            f"Shape mismatch after preprocessing: X={self.X_scaled.shape[0]}, y={self.ground_truth.shape[0]}"

    def save_figure(self, fig, prefix="plot"):
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{self.file_stem}_{timestamp}.png"
        save_path = os.path.join("output", filename)
        fig.savefig(save_path, bbox_inches='tight')

    def find_optimal_pca_components(self, max_components=100):
        max_possible_components = min(self.X_scaled.shape[1], self.X_scaled.shape[0] - 1, max_components)
        if max_possible_components < 2:
            return max_possible_components

        pca_test = PCA()
        pca_test.fit(self.X_scaled)
        cumsum = np.cumsum(pca_test.explained_variance_ratio_)
        optimal_components = np.argmax(cumsum >= 0.95) + 1
        optimal_components = min(optimal_components, max_possible_components)
        plt.figure(figsize=(10, 6))
        n_components_to_plot = min(len(cumsum), 100)
        plt.plot(range(1, n_components_to_plot + 1), cumsum[:n_components_to_plot], 'bo-')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        plt.axvline(x=optimal_components, color='g', linestyle='--', label=f'Optimal: {optimal_components}')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA: Cumulative Explained Variance')
        plt.legend()
        plt.grid(True)
        fig = plt.gcf()
        self.save_figure(fig, prefix="pca_variance")
        plt.close(fig)
        plt.show()

        return optimal_components

    def apply_dimensionality_reduction(self, method='pca', n_components=None):
        if n_components is None:
            n_components = self.find_optimal_pca_components()

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError("Method must be 'pca' or 'svd'")

        self.X_reduced = reducer.fit_transform(self.X_scaled)
        print(f"{method.upper()}-reduced shape: {self.X_reduced.shape}")

        if hasattr(reducer, 'explained_variance_ratio_'):
            print(f"Total explained variance: {np.sum(reducer.explained_variance_ratio_):.3f}")
    
    def estimate_eps(self, k=5):
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.X_reduced)
        distances, indices = nbrs.kneighbors(self.X_reduced)
        
        k_dist = np.sort(distances[:, -1])
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(k_dist)), k_dist)
        plt.axhline(y=np.mean(k_dist), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(k_dist):.3f}')
        plt.axhline(y=np.median(k_dist), color='g', linestyle='--', 
                   label=f'Median: {np.median(k_dist):.3f}')
        
        x = np.array(range(len(k_dist)))
        y = k_dist
        
        dy = np.gradient(y)
        d2y = np.gradient(dy)
        
        curvature = np.abs(d2y)
        
        window_size = max(5, int(len(curvature) * 0.01)) 
        smoothed_curvature = np.convolve(curvature, np.ones(window_size)/window_size, mode='valid')
        
        pad = window_size // 2
        smoothed_indices = np.arange(pad, len(curvature) - pad)
        
        start_idx = int(len(smoothed_indices) * 0.05)  
        end_idx = int(len(smoothed_indices) * 0.95)   
        
        valid_range = slice(start_idx, end_idx)
        elbow_idx = smoothed_indices[valid_range][np.argmax(smoothed_curvature[valid_range])]
        elbow_value = k_dist[elbow_idx]
        
        plt.axvline(x=elbow_idx, color='m', linestyle='--', 
                   label=f'Elbow point: {elbow_value:.3f}')
        
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'Distance to {k}th nearest neighbor')
        plt.title('K-Distance Graph for Eps Estimation')
        plt.legend()
        plt.grid(True)
        fig = plt.gcf()
        self.save_figure(fig, prefix="eps_estimation")
        plt.close(fig)
        plt.show()
        
        suggested_eps = elbow_value
        eps_values = [
            suggested_eps * 0.5,
            suggested_eps * 0.75,
            suggested_eps,
            suggested_eps * 1.25,
            suggested_eps * 1.5
        ]
        
        return eps_values
    
    def grid_search_dbscan(self, eps_values=None, min_samples_values=None):
        if eps_values is None:
            eps_values = self.estimate_eps()
        
        if min_samples_values is None:
            dims = self.X_reduced.shape[1]
            min_samples_values = [max(5, 2*dims), max(10, 3*dims), max(15, 4*dims)]
        param_grid = ParameterGrid({
            'eps': eps_values,
            'min_samples': min_samples_values
        })
        
        best_score = -1
        best_params = None
        best_labels = None
        best_n_clusters = 0
        best_noise_ratio = 1.0
        
        results = []
        
        total_combinations = len(param_grid)
        for i, params in enumerate(param_grid):
            print(f"Testing combination {i+1}/{total_combinations}: eps={params['eps']:.3f}, min_samples={params['min_samples']}")
            
            start_time = time.time()
            dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            labels = dbscan.fit_predict(self.X_reduced)
            end_time = time.time()
            
            if len(labels) != len(self.ground_truth):
                print(f"  → ERROR: Label shape mismatch! labels={len(labels)}, ground_truth={len(self.ground_truth)}")
                continue
            
            n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            if n_clusters < 1:
                print(f"  → No clusters found (all noise)")
                results.append({
                    'eps': params['eps'],
                    'min_samples': params['min_samples'],
                    'n_clusters': 0,
                    'noise_ratio': noise_ratio,
                    'silhouette': -1,
                    'ari': -1,
                    'nmi': -1,
                    'runtime': end_time - start_time
                })
                continue
            
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1 and n_clusters > 1:
                try:
                    silhouette = silhouette_score(
                        self.X_reduced[non_noise_mask], 
                        labels[non_noise_mask]
                    )
                except:
                    silhouette = -1
            else:
                silhouette = -1
            try:
                ari = adjusted_rand_score(self.ground_truth, labels)
                nmi = normalized_mutual_info_score(self.ground_truth, labels)
            except Exception as e:
                print(f"  → Error calculating supervised metrics: {e}")
                ari = -1
                nmi = -1
            
            print(f"  → Found {n_clusters} clusters, {noise_ratio:.2%} noise")
            print(f"  → Silhouette: {silhouette:.3f}, ARI: {ari:.3f}, NMI: {nmi:.3f}")
            
            results.append({
                'eps': params['eps'],
                'min_samples': params['min_samples'],
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'silhouette': silhouette,
                'ari': ari,
                'nmi': nmi,
                'runtime': end_time - start_time
            })
            
            score = silhouette if silhouette > 0 else (ari + nmi) / 2
            
            if hasattr(self, 'ground_truth') and n_clusters > len(np.unique(self.ground_truth)) * 2:
                score -= 0.1
            if noise_ratio > 0.5:
                score -= 0.1
            
            if score > best_score:
                best_score = score
                best_params = params
                best_labels = labels
                best_n_clusters = n_clusters
                best_noise_ratio = noise_ratio
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty and len(results_df) > 1:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            pivot_clusters = results_df.pivot_table(
                index='min_samples', columns='eps', values='n_clusters', aggfunc='first'
            )
            sns.heatmap(pivot_clusters, annot=True, cmap='viridis', fmt='d')
            plt.title('Number of Clusters')
            
            plt.subplot(2, 2, 2)
            pivot_noise = results_df.pivot_table(
                index='min_samples', columns='eps', values='noise_ratio', aggfunc='first'
            )
            sns.heatmap(pivot_noise, annot=True, cmap='coolwarm', fmt='.2%')
            plt.title('Noise Ratio')
            
            plt.subplot(2, 2, 3)
            pivot_silhouette = results_df.pivot_table(
                index='min_samples', columns='eps', values='silhouette', aggfunc='first'
            )
            sns.heatmap(pivot_silhouette, annot=True, cmap='RdYlGn', fmt='.3f')
            plt.title('Silhouette Score')
            
            plt.subplot(2, 2, 4)
            pivot_ari = results_df.pivot_table(
                index='min_samples', columns='eps', values='ari', aggfunc='first'
            )
            sns.heatmap(pivot_ari, annot=True, cmap='RdYlGn', fmt='.3f')
            plt.title('Adjusted Rand Index')
            
            plt.tight_layout()
            fig = plt.gcf()
            self.save_figure(fig, prefix="grid_search_heatmap")
            plt.show()
        
        if best_params is not None:
            print("\nBest parameters:")
            print(f"  eps = {best_params['eps']:.3f}")
            print(f"  min_samples = {best_params['min_samples']}")
            print(f"  Found {best_n_clusters} clusters with {best_noise_ratio:.2%} noise points")
            print(f"  Score: {best_score:.3f}")
            
            self.best_params = best_params
            self.best_score = best_score
            self.labels_ = best_labels
        else:
            print("No suitable parameters found")
        
        return results_df

    def fit_best_dbscan(self, eps=None, min_samples=None):
        if eps is not None and min_samples is not None:
            params = {'eps': eps, 'min_samples': min_samples}
            print(f"Using provided parameters: eps={eps:.3f}, min_samples={min_samples}")
        elif self.best_params is not None:
            params = self.best_params
            print(f"Using best parameters from grid search: eps={params['eps']:.3f}, min_samples={params['min_samples']}")
        else:
            print("No parameters provided. Running grid search...")
            self.grid_search_dbscan()
            if self.best_params is None:
                raise ValueError("Could not find suitable parameters")
            params = self.best_params
        
        dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        self.labels_ = dbscan.fit_predict(self.X_reduced)
        self.best_params = params
        
        n_clusters = len(np.unique(self.labels_)) - (1 if -1 in self.labels_ else 0)
        noise_ratio = np.sum(self.labels_ == -1) / len(self.labels_)
        print(f"DBSCAN clustering completed: {n_clusters} clusters found, {noise_ratio:.2%} noise points")
        
        if hasattr(self, 'ground_truth'):
            try:
                ari = adjusted_rand_score(self.ground_truth, self.labels_)
                nmi = normalized_mutual_info_score(self.ground_truth, self.labels_)
                print(f"Adjusted Rand Index: {ari:.3f}")
                print(f"Normalized Mutual Information: {nmi:.3f}")
            except Exception as e:
                print(f"Error calculating supervised metrics: {e}")
        
        non_noise_mask = self.labels_ != -1
        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
            try:
                silhouette = silhouette_score(
                    self.X_reduced[non_noise_mask], 
                    self.labels_[non_noise_mask]
                )
                print(f"Silhouette Score: {silhouette:.3f}")
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
        
        return self.labels_

    def visualize_results(self, perplexity=30):
        if self.labels_ is None:
            raise ValueError("No clustering results found. Run fit_best_dbscan first.")

        perplexity = min(perplexity, (len(self.X_reduced) - 1) // 3)

        print(f"Running t-SNE for visualization with perplexity={perplexity}...")
        try:
            tsne = TSNE(n_components=2, random_state=self.random_state, 
                   perplexity=perplexity, metric='euclidean', learning_rate='auto')
            X_2d = tsne.fit_transform(self.X_reduced)
        except Exception as e:
            print(f"t-SNE failed: {e}")
            print("Using PCA for 2D visualization instead...")
            pca_2d = PCA(n_components=2, random_state=self.random_state)
            X_2d = pca_2d.fit_transform(self.X_reduced)
    
        unique_clusters = np.unique(self.labels_)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
        if hasattr(self, 'ground_truth'):
            try:
                ari_score = adjusted_rand_score(self.ground_truth, self.labels_)
                nmi_score = normalized_mutual_info_score(self.ground_truth, self.labels_)
            except:
                ari_score = 0
                nmi_score = 0
        else:
            ari_score = 0
            nmi_score = 0
    
        base_name = self.file_stem
    
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.labels_, cmap='tab10', s=15, alpha=0.8)
        plt.colorbar(scatter, label="Cluster")
    
        if n_clusters > 0:
            centroids_2d = np.zeros((n_clusters, 2))
            cluster_idx = 0
            for i in unique_clusters:
                if i != -1:  
                    cluster_points = X_2d[self.labels_ == i]
                    if len(cluster_points) > 0:
                        centroids_2d[cluster_idx] = np.mean(cluster_points, axis=0)
                        cluster_idx += 1
        
            if cluster_idx > 0:  
                plt.scatter(centroids_2d[:cluster_idx, 0], centroids_2d[:cluster_idx, 1], 
                           c='black', s=100, alpha=0.8, marker='X')
    
        if -1 in unique_clusters:
            noise_points = X_2d[self.labels_ == -1]
            if len(noise_points) > 0:
                plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=15, alpha=0.5, 
                       marker='x', label='Noise')
                plt.legend()
    
        plt.title(f"t-SNE visualization of {base_name} embeddings clustered with DBSCAN\n" 
              f"eps={self.best_params['eps']:.3f}, min_samples={self.best_params['min_samples']}, "
              f"ARI={ari_score:.4f}, NMI={nmi_score:.4f}")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout()
    
        os.makedirs("output", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dbscan_plot_path = os.path.join("output", f"{base_name}_dbscan_{timestamp}.png")
        plt.savefig(dbscan_plot_path)
        plt.show()
        print(f"Saved plot to {dbscan_plot_path}")
    
        if hasattr(self, 'ground_truth'):
            plt.figure(figsize=(16, 6))
        
            plt.subplot(1, 2, 1)
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.labels_, cmap='tab10', s=15, alpha=0.8)
            if -1 in unique_clusters:
                noise_points = X_2d[self.labels_ == -1]
                if len(noise_points) > 0:
                    plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=15, alpha=0.5, 
                           marker='x', label='Noise')
                    plt.legend()
                
            plt.title(f"DBSCAN Clustering\neps={self.best_params['eps']:.3f}, "
                  f"min_samples={self.best_params['min_samples']}, ARI={ari_score:.4f}")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
        
            plt.subplot(1, 2, 2)
            plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.ground_truth, cmap='tab10', s=15, alpha=0.8)
            plt.title("Ground Truth Labels")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
        
            plt.tight_layout()
            comparison_plot_path = os.path.join("output", f"{base_name}_dbscan_groundtruth_{timestamp}.png")
            plt.savefig(comparison_plot_path)
            plt.show()
            print(f"Saved comparison plot to {comparison_plot_path}")
    
        print("\nCluster distributions:")
        for i in unique_clusters:
            count = np.sum(self.labels_ == i)
            percentage = count / len(self.labels_) * 100
            if i == -1:
                print(f"Noise points: {count} samples ({percentage:.1f}%)")
            else:
                print(f"Cluster {i}: {count} samples ({percentage:.1f}%)")
    
        if hasattr(self, 'ground_truth'):
            print("\nGround truth label distribution:")
            unique_labels, counts = np.unique(self.ground_truth, return_counts=True)
            for label, count in zip(unique_labels, counts):
                percentage = count / len(self.ground_truth) * 100
                print(f"Label {label}: {count} samples ({percentage:.1f}%)")

    def get_cluster_summary(self):
        """
        Get a summary of the clustering results
        """
        if self.labels_ is None:
            return "No clustering results available"
        
        unique_clusters = np.unique(self.labels_)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        noise_ratio = np.sum(self.labels_ == -1) / len(self.labels_)
        
        summary = {
            'n_clusters': n_clusters,
            'noise_points': np.sum(self.labels_ == -1),
            'noise_ratio': noise_ratio,
            'total_points': len(self.labels_),
            'best_params': self.best_params
        }
        
        if hasattr(self, 'ground_truth'):
            try:
                summary['ari'] = adjusted_rand_score(self.ground_truth, self.labels_)
                summary['nmi'] = normalized_mutual_info_score(self.ground_truth, self.labels_)
            except:
                summary['ari'] = None
                summary['nmi'] = None
        
        return summary

if __name__ == "__main__":
    clustering = EnhancedDBSCANClustering(random_state=42)
    clustering.load_and_preprocess_data(
        npy_path="/Users/dariganarmanova/Downloads/bert_simple.npy", 
        csv_path="/Users/dariganarmanova/cse304-term-project/data/preprocessed/bbc_encoded.csv"
    )
    clustering.apply_dimensionality_reduction(method='pca', n_components=50)
    results = clustering.grid_search_dbscan()
    clustering.visualize_results(perplexity=30)