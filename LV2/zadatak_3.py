import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2/img/road.jpg")
img = img[:,:,0].copy()

plt.figure()
plt.imshow(img,cmap="gray")
plt.show()

img_bright = img*1.5
img_bright = np.clip(img_bright,0,255)
plt.imshow(img_bright, cmap="gray")
plt.show()

height,width = img.shape
plt.imshow(img[:,width//4:width//2],cmap="gray")
plt.show()

img_rotated = np.rot90(img, -1)
plt.imshow(img_rotated, cmap="gray")
plt.show()

img_mirror = np.fliplr(img)
plt.imshow(img_mirror, cmap="gray")
plt.show()