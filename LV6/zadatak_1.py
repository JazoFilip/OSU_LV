import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("LV6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()


#Model KNN ZADATAK 1:
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n,y_train)
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("\nKNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_p_KNN)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=5), tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN)))
plt.tight_layout()
plt.show()


#ZADATAK 2
KNN_model_1 = KNeighborsClassifier(n_neighbors=1)
KNN_model_1.fit(X_train_n,y_train)
y_train_p_KNN_1 = KNN_model_1.predict(X_train_n)
y_test_p_KNN_1 = KNN_model_1.predict(X_test_n)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=1), tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN_1)))
plt.tight_layout()

KNN_model_100 = KNeighborsClassifier(n_neighbors=100)
KNN_model_100.fit(X_train_n,y_train)
y_train_p_KNN_100 = KNN_model_100.predict(X_train_n)
y_test_p_KNN_100 = KNN_model_100.predict(X_test_n)
plot_decision_regions(X_train_n, y_train, classifier=KNN_model_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=100), tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_p_KNN_100)))
plt.tight_layout()
plt.show()


#Uankrsna Validacija
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])

# hiperparametri
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [10, 1, 0.1, 0.01],
    'model__kernel': ['rbf']
}

# GridSearchCV
svm_gscv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# treniranje
svm_gscv.fit(X_train, y_train)

# rezultati
print("Najbolji parametri:", svm_gscv.best_params_)
print("Najbolja CV tocnost:", svm_gscv.best_score_)

best_model = svm_gscv.best_estimator_

y_test_pred = best_model.predict(X_test)

print("Test tocnost:", accuracy_score(y_test, y_test_pred))

#SVM model RBF kernel
svm_model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
svm_model.fit(X_train_n, y_train)

y_train_p_svm = svm_model.predict(X_train_n)
y_test_p_svm = svm_model.predict(X_test_n)

print("SVM (RBF):")
print("Tocnost train:", accuracy_score(y_train, y_train_p_svm))
print("Tocnost test:", accuracy_score(y_test, y_test_p_svm))

plot_decision_regions(X_train_n, y_train, classifier=svm_model)
plt.title("SVM RBF (C=1, gamma=0.1)")
plt.show()

parameters = [
    (0.1, 0.01),
    (1, 0.1),
    (10, 1)
]

for i, (C, gamma) in enumerate(parameters):
    model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    model.fit(X_train_n, y_train)

    plot_decision_regions(X_train_n, y_train, classifier=model)
    plt.title(f"C={C}, gamma={gamma}")

plt.tight_layout()
plt.show()

# optimalna vrijednost hiperparametra C i γ
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# pipeline: skaliranje + SVM
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf'))
])

# raspon hiperparametara
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [1, 0.1, 0.01, 0.001]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Optimalni parametri:", grid.best_params_)
print("Najbolja CV tocnost:", grid.best_score_)

best_model = grid.best_estimator_
y_test_pred = best_model.predict(X_test)

print("Test tocnost:", accuracy_score(y_test, y_test_pred))