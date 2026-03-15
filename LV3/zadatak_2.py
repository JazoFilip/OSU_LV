import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("LV3/data_C02_emission.csv")

plt.figure()
data["CO2 Emissions (g/km)"].plot(kind="hist", bins=20)

plt.figure()
for fuelType in data["Fuel Type"].unique():
    subset = data[data["Fuel Type"] == fuelType]

    plt.scatter(subset["Fuel Consumption City (L/100km)"],
                subset["CO2 Emissions (g/km)"],
                label=fuelType)
    
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos gradske potrošnje goriva i emisije CO2")
plt.legend()

data.boxplot(column=["Fuel Consumption Hwy (L/100km)"],by="Fuel Type")

plt.figure()
vehicles_by_fuelType = data.groupby("Fuel Type").size()
vehicles_by_fuelType.plot(kind="bar")
plt.xlabel("Tip goriva")
plt.ylabel("Broj vozila")
plt.title("Broj vozila po tipu goriva")

plt.figure()
avgCO2_by_cyl = data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean()
avgCO2_by_cyl.plot(kind="bar")
plt.xlabel("Broj cilindara")
plt.ylabel("Prosječni CO2")
plt.title("Prosječni CO2 po broju cilindara")

plt.show()

