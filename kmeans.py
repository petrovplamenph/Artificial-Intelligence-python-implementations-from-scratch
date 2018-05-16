import math
from statistics import mean
import matplotlib.pyplot as plt
import random
class Cluster:
    
    def __init__(self,horiz_pos, vert_pos):

        self._horiz_center = horiz_pos
        self._vert_center = vert_pos
        
    
    def horiz_center(self):
        """
        Get the averged horizontal center of cluster
        """
        return self._horiz_center
    
    def vert_center(self):
        """
        Get the averaged vertical center of the cluster
        """
        return self._vert_center
   
        
    def copy(self):
        """
        Return a copy of a cluster
        """
        copy_cluster = Cluster(self._horiz_center, self._vert_center)
        return copy_cluster


    def distance(self, other_point):
        """
        Compute the Euclidean distance between two clusters
        """
        vert_dist = self._vert_center - other_point[1]
        horiz_dist = self._horiz_center - other_point[0]
        return math.sqrt(vert_dist ** 2 + horiz_dist ** 2)
        
    def recalculate_center(self, new_points):
        x = 0
        y = 0
        for point in new_points:
            x += point[0]
            y += point[1]

        self._horiz_center = x/len(new_points)             
        self._vert_center =  y/len(new_points) 

        return
        
        
def load_data(file_name):
    with open(file_name) as f:
        data = []
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')

            data.append((float(line[0]),float(line[1])))
    return data

def load_data2(file_name):
    with open(file_name) as f:
        data = []
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            data.append((float(line[0]),float(line[1])))
    return data


def kmeans_clustering(data, num_clusters, num_iterations):
    """
    Compute the k-means clustering of a set of clusters
    Note: the function may not mutate cluster_list
    
    Input: List of clusters, integers number of clusters and number of iterations
    Output: List of clusters whose length is num_clusters
    """
    
    all_points = list(data)
    random.shuffle(all_points )
    new_clusters = list(all_points[:num_clusters])
    centers_cluster_list = []
    delta = 0.1
    for clust in new_clusters:
        centers_cluster_list.append(Cluster(clust[0],clust[1]))
    for dummy_itter in range(num_iterations):
        cluster_to_points = {}
        for clust_idx in range(num_clusters):
            cluster_to_points[clust_idx] = []
        for point in all_points:
            min_dist = float("inf")

            for clust_idx in range(num_clusters):

                dist = centers_cluster_list[clust_idx].distance(point)
                if dist < min_dist:
                    min_dist = dist
                    cluster_index = clust_idx
            cluster_to_points[cluster_index].append(point)
            #make_copy_of_cluster_ceters:
        old_clust = []
        for clust_idx in range(num_clusters):
            old_clust.append(centers_cluster_list[clust_idx].copy())
            centers_cluster_list[clust_idx].recalculate_center(cluster_to_points[clust_idx])
        Flag = True
        for clust_idx in range(num_clusters):
            x = centers_cluster_list[clust_idx].horiz_center()
            y = centers_cluster_list[clust_idx].vert_center()
            x_old = old_clust[clust_idx].horiz_center()
            y_old = old_clust[clust_idx].vert_center()
            if (abs(x - x_old) < delta) or (abs(y - y_old)< delta):
                Flag = False
                
                break
        if Flag:
            break 
    x = []
    y = []
    final_clusturs = []
    for clust_idx in range(num_clusters):
        for point in cluster_to_points[clust_idx]:
            x.append(point[0])
            y.append(point[1])
        final_clusturs += [clust_idx]*len(cluster_to_points[clust_idx])

    plt.scatter(x, y, c = final_clusturs)
    plt.show()
    return centers_cluster_list

def row_data_vis(data):
    x = []
    y = []
    for point in data:
        x.append(point[0])
        y.append(point[1])
    plt.plot(x, y, 'o')
    plt.show()
#data = load_data2('unbalance.txt')
data = load_data('normal.txt')
kmeans_clustering(data, 8, 800)
#row_data_vis(data)
       