
try:
    x = float(input("Upisite ocjenu u intervalu 0.0 i 1.0: " ))


    if (x < 0.0) or (x > 1.0):
        raise ValueError("Number not in range")

    
except ValueError as e:
    print("Greška: Unesite broj između 0.0 i 1.0.",e)

else:
    if x >= 0.9:
        print("A")
    elif x >= 0.8:
        print("B")
    elif x >= 0.7:
        print("C")
    elif x >= 0.6:
        print("D")
    else:
        print("F")