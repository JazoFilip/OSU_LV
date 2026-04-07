import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn import svm

# -------------------------------
# Funkcija za crtanje granice odluke
# -------------------------------
def plot_decision_regions(X, y, classifier, ax, title="", resolution=0.02):
    markers = ('s', 'x')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}"
        )

    ax.set_title(title)


# Učitavanje podataka
data = pd.read_csv("LV6/Social_Network_Ads.csv")

X = data[["Age", "EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()


# Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)


# Skaliranje
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)


# 1. RBF SVM - različiti C i gamma

parameters = [
    (0.1, 0.01),
    (1, 0.1),
    (10, 1),
    (100, 10)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

print("=== RBF SVM (različiti C i gamma) ===")

for i, (C, gamma) in enumerate(parameters):
    model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train_n, y_train)

    # predikcije
    y_train_pred = model.predict(X_train_n)
    y_test_pred = model.predict(X_test_n)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"C={C}, gamma={gamma}")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")

    plot_decision_regions(
        X_train_n, y_train,
        classifier=model,
        ax=axes[i],
        title=f"C={C}, gamma={gamma}"
    )

plt.tight_layout()
plt.show()



# 2. Različiti kerneli

kernels = ['linear', 'rbf', 'poly']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

print("\n=== Usporedba kernela ===")

for i, k in enumerate(kernels):
    model = svm.SVC(kernel=k, C=1, gamma=0.1)
    model.fit(X_train_n, y_train)

    y_test_pred = model.predict(X_test_n)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"Kernel={k}, Test accuracy={test_acc:.3f}")

    plot_decision_regions(
        X_train_n, y_train,
        classifier=model,
        ax=axes[i],
        title=f"Kernel={k}"
    )

plt.tight_layout()
plt.show()