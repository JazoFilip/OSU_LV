import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# ucitaj podatke
data = pd.read_csv("LV6/Social_Network_Ads.csv")

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podjela
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=10)

# pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# hiperparametri (K vrijednosti)
param_grid = {
    'knn__n_neighbors': list(range(1, 21)),
    'knn__weights': ['uniform', 'distance']
}

# GridSearchCV
knn_gscv = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# treniranje
knn_gscv.fit(X_train, y_train)

# rezultati
print("Najbolji parametri:", knn_gscv.best_params_)
print("Najbolja CV tocnost:", knn_gscv.best_score_)

# test
best_model = knn_gscv.best_estimator_
y_test_pred = best_model.predict(X_test)

print("Test tocnost:", accuracy_score(y_test, y_test_pred))