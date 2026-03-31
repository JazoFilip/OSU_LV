import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# ZADATAK a)
plt.figure()
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap="bwr",marker="o",label="Train")
plt.scatter(X_test[:,0],X_test[:,1],c=y_test,cmap="bwr",marker="x",label="Test")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prikaz podataka (train i test)')
plt.legend()

# ZADATAK b)

logRegression_model = LogisticRegression()
logRegression_model.fit(X_train,y_train)

# ZADATAK c)

theta0 = logRegression_model.intercept_[0]
theta1 = logRegression_model.coef_[0][0]
theta2 = logRegression_model.coef_[0][1]

print("theta0:",theta0)
print("theta1:",theta1)
print("theta2:",theta2)


plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], 
            c=y_train, cmap='bwr', marker='o', label='Train')
x1_vals = np.linspace(X_train[:,0].min(), X_train[:,0].max(), 100)
x2_vals = -(theta0 + theta1 * x1_vals) / theta2
plt.plot(x1_vals, x2_vals, 'k-', label='Decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistička regresija - granica odluke')
plt.legend()

# ZADATAK d)
y_test_p = logRegression_model.predict(X_test)
print ("Tocnost:", accuracy_score ( y_test , y_test_p ) )

cm = confusion_matrix(y_test,y_test_p)
print("Matrica zabune:")
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

print(classification_report(y_test,y_test_p))

# ZADATAK e)

correct = y_test == y_test_p
incorrect = y_test != y_test_p

plt.figure()
plt.scatter(X_test[correct,0], X_test[correct,1], c="green",marker="o", label="Tocno klasificirani")
plt.scatter(X_test[incorrect,0],X_test[incorrect,1],c="black",marker="x",label="Netocno klasificirani")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Rezultati klasifikacije (test skup)')
plt.legend()
plt.show()