import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])
plt.show()

# normalizacija
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza s Dropout slojevima
model = keras.Sequential()

model.add(layers.Input(shape=(32, 32, 3)))

model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # 🔹 dropout

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # 🔹 dropout

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # 🔹 dropout

model.add(layers.Flatten())

model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5))  # 🔹 dropout (jači u FC dijelu)

model.add(layers.Dense(10, activation='softmax'))

model.summary()

# TensorBoard callback (NOVI direktorij)
my_callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='logs/cnn_dropout',
        update_freq=100
    )
]

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# treniranje
model.fit(
    X_train_n,
    y_train,
    epochs=40,
    batch_size=64,
    validation_split=0.1,
    callbacks=my_callbacks
)

# evaluacija
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0 * score[1]:.2f}')

# Dodavanjem dropout slojeva smanjuje se prenaučenost (overfitting).
# Tijekom učenja točnost na skupu za učenje raste sporije nego bez dropout-a, 
# ali je razlika između točnosti na skupu za učenje i validacijskom skupu manja.
# model bolje generalizira na nove podatke.
# model ima veću točnsot na validation skupu. Najveća razlika je u grafu za loss.
# Bez dropout slojeva loss na validation skupu konstantno raste, dok kod dropout skupa pada 
# te dostiže vrijednsot oko 0.6 nakon koje stagnira.
