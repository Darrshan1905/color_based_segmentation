import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

#input the image
image = cv2.imread('./bubbles.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#reshape the image to a standard format (number of pixels as rows and 3 columns for rgb)
reshaped_image = image.reshape((-1, 3))

# Define a range of values for k
min_components = 1
max_components = 20
n_components = np.arange(min_components, max_components)

log_likelihoods = []

for k in n_components:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(reshaped_image)
    
    log_likelihoods.append(gmm.score(reshaped_image))

plt.plot(n_components, log_likelihoods, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Log Likelihood')
plt.title('Log Likelihood vs. Number of Components')
plt.grid(True)
plt.show()

# Find the optimal number of components
diffs = np.diff(log_likelihoods)
elbow_point = np.where(diffs < 0)[0][0] + 2
print("Elbow point (optimal number of components):", elbow_point)

# Segmentation using GMM with optimal K
gmm = GaussianMixture(n_components=elbow_point, random_state=42)
gmm.fit(reshaped_image)

labels = gmm.predict(reshaped_image)

segmented_image = labels.reshape(image.shape[:2])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('Segmented Image')
plt.axis('off')

plt.show()
