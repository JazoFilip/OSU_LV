import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 5)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')


km = KMeans(n_clusters=2, init="random", n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1], c=labels,cmap="viridis")
plt.title("K-means grupiranje")
plt.show()

# Rezultati grupiranja ovise o obliku distribucije podataka i udaljenosti između grupa
# Uz optimalan broj K grupa algoritam daje dobar rezultat kada su grupe podataka razdvojene i kompaktne
# Ako su podaci raspršeni i granice između grupa nisu jasne algoritam ne daje dobar rezultat jer za
# dodjelu grupa K-means algoritam koristi samo udaljenost od centroida