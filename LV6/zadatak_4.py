import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split, GridSearchCV
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


# GridSearchCV za SVM (RBF)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    svm.SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train_n, y_train)

# Rezultati
print("Najbolji parametri:", grid.best_params_)
print("Najbolja CV točnost:", grid.best_score_)

best_model = grid.best_estimator_

# Evaluacija na test skupu
y_train_pred = best_model.predict(X_train_n)
y_test_pred = best_model.predict(X_test_n)

print("\nEvaluacija najboljeg modela:")
print("Train točnost:", accuracy_score(y_train, y_train_pred))
print("Test točnost:", accuracy_score(y_test, y_test_pred))


# Prikaz granice odluke
fig, ax = plt.subplots(figsize=(6, 5))

plot_decision_regions(
    X_train_n,
    y_train,
    classifier=best_model,
    ax=ax,
    title=f"Best SVM (C={grid.best_params_['C']}, gamma={grid.best_params_['gamma']})"
)

plt.tight_layout()
plt.show()