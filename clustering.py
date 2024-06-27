import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import pdist, cdist


def calculate_dunn_index(data, labels):
    clusters = np.unique(labels)
    intra_cluster_distances = []
    inter_cluster_distances = []
    
    for cluster in clusters:
        cluster_points = data[labels == cluster]
        if len(cluster_points) > 1:
            intra_cluster_distance = np.max(pdist(cluster_points))
        else:
            intra_cluster_distance = 0
        intra_cluster_distances.append(intra_cluster_distance)
    
    
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            cluster_i_points = data[labels == clusters[i]]
            cluster_j_points = data[labels == clusters[j]]
            inter_cluster_distance = np.min(cdist(cluster_i_points, cluster_j_points))
            inter_cluster_distances.append(inter_cluster_distance)
    
    dunn_index = np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
    return dunn_index
    

    
    

class Clustering():
    def __init__(self,model='spectral') -> None:
        self.model = model
        self.inf_list = []
        self.clusters = []
        self.k = 0
    def find_best_k_cluster(self , data , k_range = (2,50) ):
        dunn_list = []
        silhouette_list = []
        
        for i in range(k_range[0] , k_range[1]):
            dic = {}

            if self.model == 'spectral':
                self.clustering_model = SpectralClustering(n_clusters=i, random_state=42 )

            elif self.model  == 'AgglomerativeClustering':
                self.clustering_model = AgglomerativeClustering(n_clusters=i)

            elif self.model  == 'kmeans':
                self.clustering_model = KMeans(n_clusters=i, random_state=42)

            
            clusters = self.clustering_model.fit_predict(data)

            dunn_avg = calculate_dunn_index(data, clusters)
            silhouette_avg = silhouette_score(data, clusters)
            dic['silhouette_avg'] = silhouette_avg
            dic['dunn_avg'] = dunn_avg
            dic['k'] = i
            dic['model'] = self.model
            dunn_list.append(dunn_avg)
            silhouette_list.append(silhouette_avg)
            self.inf_list.append(dic)
            
        print(f'finding best k cluster with model {self.model} is done')

        self.inf_list = sorted(self.inf_list, key=lambda x: (-x['silhouette_avg']))
        self.inf_df = pd.DataFrame(self.inf_list)


    def plot_best_K(self , n_clusters=5):
        list_silhouette = []
        for i in sorted(self.inf_list, key=lambda x: (x['k'])):
            list_silhouette.append(i['silhouette_avg'])

        plt.plot(range(2,50),list_silhouette, marker='o', linestyle='-', color='b', label='silhouettes')
        max_value = max(list_silhouette)
        max_index = list_silhouette.index(max_value)
        plt.axhline(y=max_value, color='r', linestyle='-', label=f'Maximum silhouette: {max_value}')
        plt.axvline(x=max_index+2, color='r', linestyle='--', label=f'K: {max_index+2}')

        plt.xlabel('K')
        plt.ylabel('silhouette')
        plt.title(f'Plot silhouette score in deferent K ({self.model})')
        plt.legend()
        plt.show()
        plt.close()


        cluster_df = pd.DataFrame(self.inf_list[ : n_clusters])
        bar_width = 0.35
        r1 = np.arange(len(cluster_df['k']))
        r2 = [x + bar_width for x in r1]

        fig, ax = plt.subplots()
        bars1 = ax.bar(r1, cluster_df['dunn_avg'], color='blue', width=bar_width, edgecolor='grey', label='Dunn Score')
        bars2 = ax.bar(r2, cluster_df['silhouette_avg'], color='green', width=bar_width, edgecolor='grey', label='Silhouette Score')


        ax.set_xlabel('k', fontweight='bold')
        ax.set_ylabel('Scores', fontweight='bold')
        ax.set_title(f' Dunn  and Silhouette vs k ({self.model})')
        ax.set_xticks([r + bar_width / 2 for r in range(len(cluster_df['k']))])
        ax.set_xticklabels(cluster_df['k'])

        ax.legend()
        plt.show()

    def fit_predict(self , k ,data):
        if self.model == 'spectral':
                self.clustering_model = SpectralClustering(n_clusters=k, random_state=42 )

        elif self.model  == 'AgglomerativeClustering':
            self.clustering_model = AgglomerativeClustering(n_clusters=k)

        elif self.model  == 'kmeans':
            self.clustering_model = KMeans(n_clusters=k, random_state=42)

        self.clusters = self.clustering_model.fit_predict(data)
        self.k = k
        
    def plot_data_with_clusters(self , data):

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=self.clusters, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'{self.model} Clustering of Leaves (k={self.k})')
        plt.show()