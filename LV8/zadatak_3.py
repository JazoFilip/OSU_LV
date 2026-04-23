import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image


model = keras.models.load_model("LV8/zadatak1.keras")

img = Image.open("LV8/test.png").convert("L")  # grayscale

img = img.resize((28, 28))  # MNIST dimenzije

img_array = np.array(img)

img_array = 255 - img_array

img_array = img_array.astype("float32") / 255

img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=-1)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

print("Predviđena znamenka:", predicted_class)
print("Vjerojatnosti:", prediction)

plt.imshow(img, cmap="gray")
plt.title(f"Predikcija: {predicted_class}")
plt.axis("off")
plt.show()