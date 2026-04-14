import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img1 = Image.imread("LV7/imgs/test_1.jpg")
img2 = Image.imread("LV7/imgs/test_2.jpg")
img3 = Image.imread("LV7/imgs/test_3.jpg")
img4 = Image.imread("LV7/imgs/test_4.jpg")
img5 = Image.imread("LV7/imgs/test_5.jpg")
img6 = Image.imread("LV7/imgs/test_6.jpg")

imgs = [img1,img2,img3,img4,img5,img6]
# prikazi originalnu sliku
# plt.figure()
# plt.title("Originalna slika")
# plt.imshow(img)
# plt.tight_layout()
# plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img1.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()


#zadatak 1:
uniqueColors = np.unique(img_array, axis=0)
print("Broj različitih boja: ",len(uniqueColors))

#zadatak 2,3,4,5
for i in range(len(imgs)):
    img = imgs[i].astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    K = 5
    kmeans = KMeans(n_clusters=K, init="random", n_init=5, random_state=0)
    kmeans.fit(img_array)
    img_array_aprox = kmeans.cluster_centers_[kmeans.labels_]
    img_aprox = np.reshape(img_array_aprox, (w,h,d))
    
    plt.subplot(1, 2, 1)
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Kvantizirana slika")
    plt.imshow(img_aprox)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
#zadatak 6
J = []
K_values = range(1, 11)

for k in K_values:
    kmeans = KMeans(n_clusters=k, init="random", n_init=5, random_state=0)
    kmeans.fit(img_array)
    J.append(kmeans.inertia_)

plt.figure()
plt.plot(K_values, J, marker='o')
plt.xlabel("Broj grupa K")
plt.ylabel("J (inertia)")
plt.title("Ovisnost J o broju grupa K")
plt.grid()
plt.show()

#zadatak 7
for k in range(K):
    mask = (kmeans.labels_ == k)   
    mask_img = np.reshape(mask, (w, h))


    plt.subplot(2,3,k+1)
    plt.title(f"Grupa {k}")
    plt.imshow(mask_img, cmap='gray')
    plt.axis('off')

plt.show()
 