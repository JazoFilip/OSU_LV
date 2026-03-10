import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2/img/road.jpg")
img = img[:,:,0].copy()

plt.figure()


plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray",alpha=0.8)


height,width = img.shape
plt.subplot(2, 2, 2)
plt.imshow(img[:,3*width//4:width//4:-1],cmap="gray")


img_rotated = np.rot90(img, -1)
plt.subplot(2, 2, 3)
plt.imshow(img_rotated, cmap="gray")


img_mirror = np.fliplr(img)
plt.subplot(2, 2, 4)
plt.imshow(img_mirror, cmap="gray")
plt.show()