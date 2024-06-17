from PIL import Image
import os
import numpy as np 
import pandas as pd 
import torchvision.transforms as transforms
from torchvision.models import resnet50
import random
import torch
import matplotlib.pyplot as plt 
import cv2
from scipy.signal import savgol_filter
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from skimage import  exposure
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, cdist



def extract_number(folder_name):
    return int(folder_name.split('.')[0])

def extract_number_files(folder_name):
    return int(folder_name.split('EX')[1].split('.')[0])

def get_images_path_list(input_folder) :
    path_list = []
    parent_folder_path = input_folder
    folders = [f for f in os.listdir(parent_folder_path) if os.path.isdir(os.path.join(parent_folder_path, f))]
    sorted_folders = sorted(folders, key=extract_number)

    for dirs in sorted_folders:
        files = os.listdir(parent_folder_path+dirs)
        file_names = [f for f in files if os.path.isfile(os.path.join(parent_folder_path+dirs, f))]
        file_names = sorted(file_names, key=extract_number_files)
        # print(dirs)
        for filename in file_names:
            image_path = os.path.join(parent_folder_path+dirs, filename)
            # print(image_path)
            path_list.append(image_path)

    return path_list


def apply_pca(data,name, n_components=2):

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)

    pca_columns = [f'PC_{name}_{i+1}' for i in range(principal_components.shape[1])] # for i in range(n_components)

    return pd.DataFrame(data=principal_components, columns=pca_columns)

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

def plot_hog_example(image):
    fd, hog_image = hog(image, orientations=8 , pixels_per_cell=(16, 16) ,cells_per_block=(1, 1),
                    visualize=True, channel_axis=-1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def plot_five_ex_clusters(clusters , images):

    # 5. Filter out noise and small clusters
    unique_clusters = np.unique(clusters)
    filtered_clusters = [cluster for cluster in unique_clusters if cluster != -1]

    # 6. Select representative images from each cluster
    n_examples = 5
    example_images = {cluster: [] for cluster in filtered_clusters}

    for cluster in filtered_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 4:  # Minimum cluster size threshold
            selected_indices = random.sample(list(cluster_indices), min(n_examples, len(cluster_indices)))
            example_images[cluster] = selected_indices

    # 7. Plot example images from each cluster
    plt.figure(figsize=(15, len(example_images) * 3))
    for i, (cluster, indices) in enumerate(example_images.items()):
        for j, index in enumerate(indices):
            plt.subplot(len(example_images), n_examples, i * n_examples + j + 1)
            plt.imshow(images[index], cmap=plt.cm.gray)
            plt.axis('off')
            if j == 0:
                plt.title(f'Cluster {cluster}')
    plt.tight_layout()
    plt.show()

def Calculate_avg_leaf_to_background_ratio(binary_images):

    black_pixels = np.sum(binary_images == 0)
    white_pixels = np.sum(binary_images == 255)
    average_leaf_to_background_ratio = white_pixels /  black_pixels if black_pixels != 0 else float('nan')
    return average_leaf_to_background_ratio

def find_Length_and_width(image_array):
    y_start = 0
    x_start = 0
    y_end , x_end = image_array.shape
    
    # find y_start , y_end
    for i in range(image_array.shape[0]): 
        if (image_array[i].sum() > 0) and (y_start == 0):
            y_start = i

        if (image_array[-i].sum()>0) and (y_end == image_array.shape[0]):
            y_end = image_array.shape[0] - i
    # find x_start , x_end
    for j in range(image_array.shape[1]): 
        if (image_array[: , j].sum() > 0) and (x_start == 0):
            x_start = j
        if (image_array[: , -j].sum() > 0) and (x_end == image_array.shape[1]):
            x_end = image_array.shape[1] - j

    return y_start , y_end , x_start , x_end

    

        
class Images_class():
    def __init__(self) -> None:
        self.cnn_model = resnet50(pretrained=True)
        self.rgb_images = []
        self.gray_images = []
        self.binary_images = []
        self.contours_images=[]
        self.list_distances=[]
        self.pil_images = []
        self.cnn_feature = np.array([])
        self.list_images_hog= np.array([])

    def get_images_inf(self , input_folder ):
        path_list = get_images_path_list(input_folder )
        for image_path in path_list:
                
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cnt = tuple()
            for i in range(len(contours)):
                if len(cnt) < len(contours[i]):
                    cnt = contours[i]

            img = Image.open(image_path).convert('RGB')
            # img = img.resize(target_size)

            self.pil_images.append(img)
            self.rgb_images.append(image)
            self.gray_images.append(gray)
            self.binary_images.append(thresholded)
            self.contours_images.append(cnt)

    def extract_cnn_features(self, target_size:tuple=(192,144)):

        model = self.cnn_model.eval()  # Set to evaluation mode
        # Define the transformation for preprocessing images
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])

        
        features = []
        for img in self.pil_images:
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feature = model(img_tensor).squeeze().numpy()
            features.append(feature)


        self.cnn_feature = np.array(features)

    def Calculate_distance_each_contour_from_centroid(self):
        for cnt in self.contours_images:
            M = cv2.moments(cnt)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

            # Calculate distance of each contour point from the centroid
            distances = []
            for point in cnt:
                x, y = point[0]
                distance = np.sqrt((x - centroid_x) ** 2 + (y - centroid_y) ** 2)
                distances.append(distance)

            # Convert to a numpy array and normalize
            distances = np.array(distances)
            distances = distances / np.max(distances)

            self.list_distances.append(list(distances))

    
    def get_pictures_hog(self):
        hog_list = []
        for image in self.rgb_images:
            fd , hog_image = hog(image, orientations=8 , pixels_per_cell=(16, 16) ,cells_per_block=(1, 1),visualize=True, channel_axis=-1)
            hog_list.append(fd)

        self.list_images_hog = np.vstack(hog_list)
