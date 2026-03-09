import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("LV2/img/road.jpg")
img = img[:,:,0].copy()
print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img,cmap="gray")   

plt.plot([100,400],[200,200], color="red",linestyle="--", linewidth=5)
plt.plot([100,500], [300,300], color="green", linestyle=":", linewidth=2, marker="o")

plt.show()