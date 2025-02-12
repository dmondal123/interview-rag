
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import faiss
import time
from mpl_toolkits.mplot3d import Axes3D

def generate_sample_vectors(n_vectors=100, n_dimensions=3, seed=42):
    """Generate synthetic vectors for demonstration."""
    np.random.seed(seed)
    vectors = np.random.randn(n_vectors, n_dimensions)
    normalized_vectors = normalize(vectors, norm='l2')
    return normalized_vectors

def plot_vectors_3d(vectors, query_vector=None, matches=None, title="Vector Space Visualization", filename=None):
    """Basic 3D visualization without regions."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all vectors
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c='blue', alpha=0.5, label='Database vectors')
    
    # Plot query vector if provided
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=100, label='Query vector')
    
    # Plot matches if provided
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    if filename:
        plt.savefig(filename, format='png')
    plt.show()

def plot_vectors_with_regions(vectors, centroids, query_vector=None, matches=None, 
                            searched_regions=None, title="Vector Space with FAISS Regions", filename=None):
    """
    Visualize vectors in 3D space with their clusters/regions.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assign each vector to nearest centroid
    distances, assignments = compute_vector_assignments(vectors, centroids)
    
    # Plot vectors colored by their cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i in range(len(centroids)):
        cluster_vectors = vectors[assignments == i]
        if len(cluster_vectors) > 0:
            # Make vectors transparent if their region wasn't searched
            alpha = 1.0 if searched_regions is None or i in searched_regions else 0.1
            ax.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], cluster_vectors[:, 2], 
                      c=[colors[i]], alpha=alpha, label=f'Region {i}') 
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
              c='black', s=100, marker='*', label='Region Centers')
    
    # Plot query vector
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=200, marker='x', label='Query Vector')
    
    # Plot matches
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    if filename:
        plt.savefig(filename, format='png')
    plt.show()

def compute_vector_assignments(vectors, centroids):
    """Compute which vectors belong to which centroids."""
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(centroids)
    distances, assignments = index.search(vectors, 1)
    return distances, assignments.ravel()

def train_kmeans_get_centroids(vectors, n_clusters):
    """Train k-means and get centroids."""
    kmeans = faiss.Kmeans(d=vectors.shape[1], k=n_clusters, niter=20, verbose=False)
    kmeans.train(vectors)
    return kmeans.centroids

def brute_force_cosine_search(database_vectors, query_vector, k=5):
    """Perform brute force cosine similarity search."""
    start_time = time.time()
    similarities = np.dot(database_vectors, query_vector)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    end_time = time.time()
    return top_k_indices, end_time - start_time

def faiss_flat_l2_search(database_vectors, query_vector, k=5):
    """Perform basic FAISS L2 search (no regions)."""
    dimension = database_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    index.add(database_vectors)
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    end_time = time.time()
    
    return indices[0], end_time - start_time



def faiss_ivf_search_realistic(database_vectors, query_vectors, k=5, n_regions=10, nprobe=3):
    """
    More realistic FAISS IVF search that:
    - Separates training time from search time
    - Handles batch queries
    
    Consider changing function name later as this was named while experimenting with various scenarios
    """
    dimension = database_vectors.shape[1]
    
    # Create and train index (this would normally be done once and saved)
    print("Training index (this is usually done once)...")
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2)
    
    train_start = time.time()
    index.train(database_vectors)
    train_time = time.time() - train_start
    
    # Add vectors (this is also usually done once)
    add_start = time.time()
    index.add(database_vectors)
    add_time = time.time() - add_start
    
    # Set number of regions to search
    index.nprobe = nprobe
    
    # Actual search (this is what we'd do many times)
    search_start = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - search_start
    
    return indices, search_time, train_time, add_time
# Code for generating data and saving each plot

n_vectors = 1000
n_dimensions = 3
k = 5

# Generate vectors
database_vectors = generate_sample_vectors(n_vectors, n_dimensions)

# Query vector
query_vector = generate_sample_vectors(1, n_dimensions)[0]

# 1. Initial Vector Space Visualization
plot_vectors_3d(database_vectors, query_vector, title="Initial Vector Space", filename="initial_vector_space.png")

# 2. Perform Brute Force Cosine Similarity Search and Save Plot
cosine_matches, _ = brute_force_cosine_search(database_vectors, query_vector, k)
plot_vectors_3d(database_vectors, query_vector, cosine_matches, title="Brute Force Cosine Similarity Results", filename="brute_force_cosine_results.png")

# 3. Perform FAISS Flat L2 Search and Save Plot
faiss_matches, _ = faiss_flat_l2_search(database_vectors, query_vector, k)
plot_vectors_3d(database_vectors, query_vector, faiss_matches, title="FAISS L2 Search Results", filename="faiss_l2_results.png")

def time_search_methods(vectors, query_vector, k):
    """Measure the time taken for Brute Force, FAISS Flat, and FAISS IVF searches."""
    # Brute Force
    start = time.time()
    brute_force_cosine_search(vectors, query_vector, k)
    brute_force_time = time.time() - start
    
    # FAISS Flat L2
    start = time.time()
    faiss_flat_l2_search(vectors, query_vector, k)
    faiss_flat_time = time.time() - start
    
    # FAISS IVF
    n_regions = 10
    nprobe = 3
    centroids = train_kmeans_get_centroids(vectors, n_regions)
    start = time.time()
    faiss_ivf_search_realistic(vectors, query_vector.reshape(1, -1), k, n_regions, nprobe)
    faiss_ivf_time = time.time() - start
    
    return brute_force_time, faiss_flat_time, faiss_ivf_time

# Timing experiments for different dataset sizes
sizes = [100, 1000, 5000, 10000, 50000]
results = {"sizes": sizes, "brute_force": [], "faiss_flat": [], "faiss_ivf": []}

for size in sizes:
    print(f"Timing searches for dataset size: {size}")
    vectors = generate_sample_vectors(size, n_dimensions=3)
    query_vector = generate_sample_vectors(1, n_dimensions=3)[0]
    
    # Measure search times
    brute_force_time, faiss_flat_time, faiss_ivf_time = time_search_methods(vectors, query_vector, k)
    results["brute_force"].append(brute_force_time)
    results["faiss_flat"].append(faiss_flat_time)
    results["faiss_ivf"].append(faiss_ivf_time)

# Search Time Comparison Plot
plt.figure(figsize=(12, 6))
plt.plot(results['sizes'], results['brute_force'], 'o-', label='Brute Force Cosine')
plt.plot(results['sizes'], results['faiss_flat'], 's-', label='FAISS Flat L2')
plt.plot(results['sizes'], results['faiss_ivf'], '^-', label='FAISS IVF')

plt.xscale('log')
plt.yscale('log')

plt.xlabel('Number of Vectors')
plt.ylabel('Average Search Time per Query (seconds)')
plt.title('Search Time Comparison: Brute Force vs FAISS Methods')
plt.grid(True)
plt.legend()
plt.savefig("search_time_comparison.png", format='png')
plt.show()
