import numpy as np
import matplotlib.pyplot as plt

rectangle_black = np.zeros((50,50))
rectangle_white = np.ones((50,50))
rectangle_first_row = np.hstack((rectangle_black,rectangle_white))
rectangle_second_row = np.fliplr(rectangle_first_row)
rectangle = np.vstack((rectangle_first_row,rectangle_second_row))

plt.figure()
plt.imshow(rectangle,cmap="gray")
plt.show()