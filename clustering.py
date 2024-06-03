from PIL import Image
import os
import numpy as np 
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
        print(dirs)
        for filename in file_names:
            image_path = os.path.join(parent_folder_path+dirs, filename)
            print(image_path)
            path_list.append(image_path)

    return path_list


def plot_five_ex_clusters(clusters , images):

    # 5. Filter out noise and small clusters
    unique_clusters = np.unique(clusters)
    filtered_clusters = [cluster for cluster in unique_clusters if cluster != -1]

    # 6. Select representative images from each cluster
    n_examples = 5
    example_images = {cluster: [] for cluster in filtered_clusters}

    for cluster in filtered_clusters:
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) > 10:  # Minimum cluster size threshold
            selected_indices = random.sample(list(cluster_indices), min(n_examples, len(cluster_indices)))
            example_images[cluster] = selected_indices

    # 7. Plot example images from each cluster
    plt.figure(figsize=(15, len(example_images) * 3))
    for i, (cluster, indices) in enumerate(example_images.items()):
        for j, index in enumerate(indices):
            plt.subplot(len(example_images), n_examples, i * n_examples + j + 1)
            plt.imshow(images[index])
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

    

    
class pythorch_images():
    def __init__(self) -> None:
        self.model = resnet50(pretrained=True)
        self.images = []
        self.features = np.array([])

    def extract_features_cnn(self):
        # model = 
        model = self.model.eval()  # Set to evaluation mode
        # Define the transformation for preprocessing images
        preprocess = transforms.Compose([
            # transforms.Resize((192,144)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Extract features from the images
        features = []
        for img in self.images:
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                feature = model(img_tensor).squeeze().numpy()
            features.append(feature)

        # return np.array(features)
        self.features = np.array(features)

    def load_images(self ,image_dir : str, target_size: tuple =(192,144) ) ->list:
        images = []
        images_path = get_images_path_list(image_dir)
        for img_path in images_path:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(target_size)
            images.append(img)

        self.images = images
        
class cv2_images():
    def __init__(self) -> None:
        self.rgb_images = []
        self.gray_images = []
        self.binary_images = []
        self.contours_images=[]
        self.list_distances=[]
        self.list_images_hog=[]

    def get_images_inf(self , input_folder):
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

            self.rgb_images.append(image)
            self.gray_images.append(gray)
            self.binary_images.append(thresholded)
            self.contours_images.append(cnt)

    def Calculate_distance_each_contour_from_centroid(self, savgol_filter_:bool=False , window_length=50 , polyorder=2):
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
            if savgol_filter_:
                smoothed_signal = savgol_filter(distances, window_length=window_length, polyorder=polyorder)
                self.list_distances.append(list(smoothed_signal))
            else:
                self.list_distances.append(list(distances))
        # return distances
    
    def get_pictures_hog(self):
        for image in self.rgb_images:
            fd , hog_image = hog(image, orientations=8 , pixels_per_cell=(16, 16) ,cells_per_block=(1, 1),visualize=True, channel_axis=-1)
            self.list_images_hog.append(fd)

    
# if __name__ == 'main':
pytch = pythorch_images()
c_v2 = cv2_images()

c_v2.get_images_inf('test/')

# pytch.load_images("test/")

# pytch.extract_features_cnn()

# print(pytch.features.shape)
# print(len(pytch.images))
c_v2.Calculate_distance_each_contour_from_centroid()
print(len(c_v2.list_distances))

plt.imshow(c_v2.binary_images[0], cmap=plt.cm.gray)
plt.show()
plt.plot(c_v2.list_distances[2])
plt.show()


kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(np.vstack(c_v2.list_images_hog)) 
print(clusters)

