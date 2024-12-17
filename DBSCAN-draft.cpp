#include <iostream>
#include <vector>
#include <cmath>
#include <set>

using namespace std;

// Function to calculate Euclidean distance between two points
double euclideanDistance(const vector<double>& point1, const vector<double>& point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return sqrt(sum);
}

// DBSCAN Algorithm
class DBSCAN {
public:
    DBSCAN(double eps, int minPts) : eps(eps), minPts(minPts), clusterID(0) {}

    // Runs the DBSCAN algorithm
    vector<int> run(const vector<vector<double>>& data) {
        labels.assign(data.size(), -1); // Initially, all points are marked as noise (-1)

        for (size_t i = 0; i < data.size(); ++i) {
            if (labels[i] != -1) continue; // Skip already visited points

            vector<int> neighbors = regionQuery(data, i);
            if (neighbors.size() < minPts) {
                labels[i] = -1; // Mark as noise
            } else {
                clusterID++; // Start a new cluster
                expandCluster(data, i, neighbors);
            }
        }

        return labels; // Return the cluster labels
    }

private:
    double eps;
    int minPts;
    int clusterID;
    vector<int> labels;

    // Expand cluster recursively for a core point
    void expandCluster(const vector<vector<double>>& data, int pointIdx, vector<int>& neighbors) {
        labels[pointIdx] = clusterID;

        size_t i = 0;
        while (i < neighbors.size()) {
            int neighborIdx = neighbors[i];

            if (labels[neighborIdx] == -1) { // If it is marked as noise, include it in the cluster
                labels[neighborIdx] = clusterID;
            }

            if (labels[neighborIdx] == -1 || labels[neighborIdx] == 0) {
                labels[neighborIdx] = clusterID;

                vector<int> newNeighbors = regionQuery(data, neighborIdx);
                if (newNeighbors.size() >= minPts) {
                    neighbors.insert(neighbors.end(), newNeighbors.begin(), newNeighbors.end());
                }
            }

            i++;
        }
    }

    // Find all neighbors within eps distance of a point
    vector<int> regionQuery(const vector<vector<double>>& data, int pointIdx) {
        vector<int> neighbors;
        for (size_t i = 0; i < data.size(); ++i) {
            if (euclideanDistance(data[pointIdx], data[i]) <= eps) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }
};

int main() {
    // Example Dataset (2D Points)
    vector<vector<double>> data = {
        {1.0, 2.0}, {2.0, 2.0}, {2.0, 3.0}, {8.0, 7.0},
        {8.0, 8.0}, {25.0, 80.0}, {8.0, 6.0}, {7.0, 7.0}
    };

    // DBSCAN Parameters
    double eps = 2.0; // Radius of neighborhood
    int minPts = 2;   // Minimum points to form a cluster

    // Run DBSCAN
    DBSCAN dbscan(eps, minPts);
    vector<int> labels = dbscan.run(data);

    // Print Cluster Labels
    cout << "Cluster Labels:" << endl;
    for (size_t i = 0; i < labels.size(); ++i) {
        cout << "Point " << i << ": Cluster " << labels[i] << endl;
    }

    return 0;
}
