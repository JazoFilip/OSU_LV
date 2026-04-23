import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# Model 
model = keras.models.load_model("LV8/zadatak1.keras")

# train i test podaci
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_test_s = x_test.astype("float32") / 255
# slike trebaju biti (28, 28, 1)
x_test_s = np.expand_dims(x_test_s, -1)

# 4. Predikcija
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# 5. Nađi pogrešne klasifikacije
wrong_indices = np.where(y_pred_classes != y_test)[0]

print(f"Ukupno pogrešno klasificiranih: {len(wrong_indices)}")

# 6. Prikaz nekoliko pogrešnih slika
num_images = 9  

plt.figure(figsize=(8, 6))

for i in range(num_images):
    idx = wrong_indices[i]
    
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    
    true_label = y_test[idx]
    predicted_label = y_pred_classes[idx]
    
    plt.title(f"Stvarna: {true_label}, Pred: {predicted_label}")
    plt.axis("off")

plt.tight_layout()
plt.show()