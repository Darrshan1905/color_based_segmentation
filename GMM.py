import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

image = cv2.imread('./bubbles.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

reshaped_image = image.reshape((-1, 3))

num_clusters = 5

gmm = GaussianMixture(n_components=num_clusters, random_state=42)
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
