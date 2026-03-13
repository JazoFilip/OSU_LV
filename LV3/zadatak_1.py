import pandas as pd

data = pd.read_csv("LV3/data_C02_emission.csv")

print(f"DataFrame sadrži {len(data)} mjerenja.\n")

print("Tipovi stupaca:")
print(data.dtypes)
print("\nbroj null vrijednosti po stupcima:")
print(data.isnull().sum())
data = data.dropna()
print(f"\nBroj dupliciranih redaka: {data.duplicated().sum()}")
data = data.drop_duplicates()
data = data.reset_index(drop=True)

for col in data.select_dtypes(include="object").columns:
     data[col] = data[col].astype("category")



print("\nTipovi stupaca nakon konverzije:")
print(data.dtypes)


print("\nb) zadatak:")
print(data.nlargest(3,["Fuel Consumption City (L/100km)"])[["Make","Model", "Fuel Consumption City (L/100km)"]])
print(data.nsmallest(3,["Fuel Consumption City (L/100km)"])[["Make","Model", "Fuel Consumption City (L/100km)"]])


print("\nc) zadatak:")
print(data[(data["Engine Size (L)"] > 2.5) & (data["Engine Size (L)"] < 3.5)]["CO2 Emissions (g/km)"].mean())

print("\nd) zadatak:")
data_Audi = data["Make"] == "Audi"
print(f"{len(data[data_Audi])} mjerenja se odnosi na vozila marke Audi.")
data_Audi_4cyl_meanC02 = data[data_Audi & (data["Cylinders"] == 4)]["CO2 Emissions (g/km)"].mean()
print(f"prosjecna emisija C02 plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara je {data_Audi_4cyl_meanC02}")

print("\ne) zadatak:")

data_with_4_6_8_cyl = data[(data.Cylinders == 4) | (data.Cylinders == 6) | (data.Cylinders == 8)]
data_with_4_6_8_cyl = data_with_4_6_8_cyl.groupby("Cylinders")
print(data_with_4_6_8_cyl["CO2 Emissions (g/km)"].mean())

print("\nf) zadatak:")

data_X_D = data[(data["Fuel Type"] == 'X') | (data["Fuel Type"] == 'D')]
data_X_D = data_X_D.groupby("Fuel Type")
print("Prosjecna gradska potrošnja u slucaju vozila koja koriste dize i regularni benzin")
print(data_X_D["Fuel Consumption City (L/100km)"].mean())

print("\ng) zadatak:")

print("Vozilo s 4 cilindra sa dizelskim motorom koja ima najvecu gradsku potrosnju goriva:")
data_4cyl_D = data[(data["Cylinders"] == 4) & (data["Fuel Type"] == 'D')]
print(data_4cyl_D[data_4cyl_D["Fuel Consumption City (L/100km)"] == data_4cyl_D["Fuel Consumption City (L/100km)"].max()])

print("\nh) zadatak:")

print(f"Rucni tip mjenjaca ima: {data["Transmission"].str.startswith('M').sum()} vozila.")

print("\ni) zadatak:")

correlation = data.corr(numeric_only=True)
pd.set_option('display.max_columns', None)
print(correlation)