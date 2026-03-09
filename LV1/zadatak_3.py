numberList = []

while(True):
    x = input("Unesite broj ili 'Done' za kraj:\n")
    if x == "Done":
        break
    try:
        x = float(x)
    except ValueError as e:
        print("Unesite broj!",e)
    else:
        numberList.append(x)

if(len(numberList) != 0):
    numberList.sort()
    print(numberList)
    print("Length: ",len(numberList))
    print("Max: ", max(numberList))
    print("Min: ", min(numberList))
    print("Mean: ", sum(numberList)/len(numberList))


