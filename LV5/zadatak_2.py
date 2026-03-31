import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, classification_report
from sklearn . metrics import accuracy_score

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("LV5/penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# ZADATAK a)

Y_train_classes, y_train_count = np.unique(y_train,return_counts=True)
y_test_classes, y_test_count = np.unique(y_test, return_counts=True)

class_names = [labels[int(c)] for c in Y_train_classes]
x = np.arange(len(class_names))

width = 0.35

plt.bar(x - width/2, y_train_count, width, label='Train')
plt.bar(x + width/2, y_test_count, width, label='Test')


plt.xticks(x, class_names)
plt.xlabel('Vrsta pingvina')
plt.ylabel('Broj primjera')
plt.title('Raspodjela klasa u train i test skupu')
plt.legend()


# ZADATAK b)

logRegression_model = LogisticRegression()
logRegression_model.fit(X_train,y_train)

# ZADATAK c)

theta0 = logRegression_model.intercept_
theta = logRegression_model.coef_


print("theta 0 za svaku klasu:",theta0)
print("Koeficijenti za svaku klasu po retcima:")
print(theta)


# ZADATAK d)

plot_decision_regions(X_train,y_train.ravel(),logRegression_model)
plt.xlabel('bill_length_mm')
plt.ylabel('flipper_length_mm')
plt.legend()




# ZADATAK e)

y_test_p = logRegression_model.predict(X_test)
print ("Tocnost:", accuracy_score ( y_test , y_test_p ) )

cm = confusion_matrix(y_test,y_test_p)
print("Matrica zabune:")
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

print(classification_report(y_test,y_test_p))

# ZADTAK f)

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                   "bill_depth_mm",
                    'flipper_length_mm',]

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

logRegression_model = LogisticRegression()
logRegression_model.fit(X_train,y_train)

theta0 = logRegression_model.intercept_
theta = logRegression_model.coef_


print("theta 0 za svaku klasu za 3 ulaza:",theta0)
print("Koeficijenti za svaku klasu po retcima za 3 ulaza:")
print(theta)

y_test_p = logRegression_model.predict(X_test)
print ("Tocnost:", accuracy_score ( y_test , y_test_p ) )

cm = confusion_matrix(y_test,y_test_p)
print("Matrica zabune za 3 ulaza:")
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()

print(classification_report(y_test,y_test_p))

plt.show()

