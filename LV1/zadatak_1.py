def total_euro( n: float, x: float): 
    return x*n

print("Unesite broj radnih sati: ")
n = float(input())
print("Unesite koliko ste placeni po radnom satu: ")
x = float(input())

print("Radni sati:", n, "h")
print("eura/h: ", x)
print(f"Ukupno: {total_euro(n,x):.2f}")

