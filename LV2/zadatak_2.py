import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("LV2/csv/data.csv",delimiter=',',skip_header=1)
print(f"Mjerenja su izvršena na {data.shape[0]} osoba")

gender = data[:,0]
height = data[:,1]
weight = data[:,2]


plt.scatter(height[::50],weight[::50],marker='.')
plt.xlabel("height")
plt.ylabel("weight")
plt.show()

print(f"Najmanja osoba: {height.min()}")
print(f"Najvisa osoba: {height.max()}")
print(f"Prosjecna visina: {height.mean()}")


ind_m = (data[:,0] == 1)
height_m = height[ind_m]
print("Muskarci:")
print(f"Najmanja visina: {height_m.min()}")
print(f"Najveca visina: {height_m.max()}")
print(f"Prosjecna visina: {height_m.mean()}")

ind_z = (data[:,0] == 0)
height_z = height[ind_z]
print("Zene:")
print(f"Najmanja visina: {height_z.min()}")
print(f"Najveca visina: {height_z.max()}")
print(f"Prosjecna visina: {height_z.mean()}")