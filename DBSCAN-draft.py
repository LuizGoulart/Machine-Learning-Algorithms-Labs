import numpy as np

def dbscan(data, eps, minPts):
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
    
    Parameters:
        data (numpy.ndarray): 2D array of data points [n_samples, n_features]
        eps (float): Radius of neighborhood
        minPts (int): Minimum number of points to form a core point

    Returns:
        labels (list): Cluster labels for each point (-1 represents noise)
    """
    # Initialize
    labels = [-1] * len(data)  # Initially, all points are labeled as noise (-1)
    cluster_id = 0
    
    def region_query(point_idx):
        """Find all points within eps distance of a given point."""
        neighbors = []
        for idx, other_point in enumerate(data):
            if np.linalg.norm(data[point_idx] - other_point) <= eps:
                neighbors.append(idx)
        return neighbors
    
    def expand_cluster(point_idx, neighbors, cluster_id):
        """Expand the cluster by recursively adding density-reachable points."""
        labels[point_idx] = cluster_id  # Assign current point to the cluster
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:  # If it's noise, make it part of the cluster
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:  # If it hasn't been visited yet
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= minPts:  # Add neighbors if it's a core point
                    neighbors += new_neighbors
            i += 1
    
    for point_idx in range(len(data)):
        if labels[point_idx] != -1:  # Skip visited points
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < minPts:
            labels[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1  # Start a new cluster
            expand_cluster(point_idx, neighbors, cluster_id)
    
    return labels

# Example Usage
if __name__ == "__main__":
    # Example 2D dataset
    data = np.array([
        [1, 2], [2, 2], [2, 3], [8, 7],
        [8, 8], [25, 80], [8, 6], [7, 7]
    ])

    # DBSCAN parameters
    eps = 2  # Neighborhood radius
    minPts = 2  # Minimum points to form a cluster

    # Run DBSCAN
    cluster_labels = dbscan(data, eps, minPts)

    # Display results
    print("Cluster labels:", cluster_labels)
