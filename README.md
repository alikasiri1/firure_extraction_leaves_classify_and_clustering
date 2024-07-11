# Leaf Classification and Clustering Project

## Overview

This project is developed as a part of the Machine Learning Fundamentals course at the University of Isfahan. The main objective is to build a classification and clustering model for leaf images.

## Author
- **Name:** Ali Kasiri

## Contents

- [Initial Processing](#initial-processing)
  - [Background Removal and Conversion](#background-removal-and-conversion)
- [Feature Extraction](#feature-extraction)
  - [Distance Features](#distance-features)
  - [Leaf Dimensions Features](#leaf-dimensions-features)
  - [Leaf Color Features](#leaf-color-features)
  - [HOG Features](#hog-features)
  - [Feature Extraction with RESNET-50](#feature-extraction-with-resnet-50)
- [Leaf Classification](#leaf-classification)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Leaf Clustering](#leaf-clustering)
  - [Feature Selection and Normalization](#feature-selection-and-normalization)
  - [Dimensionality Reduction](#dimensionality-reduction)
  - [Model Evaluation Metrics](#model-evaluation-metrics)
  - [Clustering Model Selection](#clustering-model-selection)
- [References](#references)

## Initial Processing

### Background Removal and Conversion

Initially, the leaf images had various backgrounds which interfered with image processing. The background was removed using the `rembg` library and converted to a uniform white background.

## Feature Extraction

Feature extraction is performed using the `feature_extraction.py` script, which includes a class `Images_class` for processing images using `cv2`.

### Distance Features

The Euclidean distance from the centroid of the leaf to its contour points is calculated and normalized. This helps in identifying leaves with different shapes and petal counts.

### Leaf Dimensions Features

- **Length to Width Ratio:** Calculated using the bounding rectangle around the leaf.
- **Leaf to Background Ratio:** The ratio of the leaf area to its bounding rectangle area.
- **Form Factor:** Measures how similar the shape is to a perfect circle.
- **Convexity:** Ratio of the perimeter of the shape to its convex hull.
- **Perimeter:** Length of the contour defining the leaf.

### Leaf Color Features

Mean color values for red, green, and blue channels are calculated and stored.

### HOG Features

Histogram of Oriented Gradients (HOG) features are extracted using the `hog` function from `skimage.feature`.

### Feature Extraction with RESNET-50

Features are extracted from images using a pre-trained ResNet-50 model.

## Leaf Classification

### Data Preprocessing

The dataset is split into training and testing sets, and features are scaled.

### Model Selection

Various classifiers are tested, including Logistic Regression, Random Forest, etc. Logistic Regression performed the best.

### Model Training

Logistic Regression is selected for final training.

### Model Evaluation

The model is evaluated using a 25-fold cross-validation, achieving an average accuracy of 93.29%.

## Leaf Clustering

### Feature Selection and Normalization

All relevant features are selected and normalized for clustering.

### Dimensionality Reduction

T-SNE is used for dimensionality reduction to two dimensions.

### Model Evaluation Metrics

Silhouette score and Dunn index are used to evaluate the clustering models.

### Clustering Model Selection

Models like K-Means, Spectral Clustering, and Agglomerative Clustering are tested, and the best model is selected based on evaluation metrics.

## References

- `rembg` library
- `cv2` library
- `scipy.signal` for Savitzky-Golay filter
- `skimage.feature` for HOG features
- Pre-trained ResNet-50 model from `tensorflow.keras.applications`
