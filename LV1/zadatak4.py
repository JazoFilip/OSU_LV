dictionary = dict()
wordList = []

try:
    fhand = open("LV1/textFiles/song.txt")
    for line in fhand:
        line = line.rstrip().lower()
        words = line.split()

        for word in words:
            
            dictionary[word] = dictionary.get(word,0) + 1

    fhand.close()

    for key,value in dictionary.items():
        if(value == 1):
            wordList.append(key)

    print(f"Ukupno riječi koje se pojavljuju samo jednom: {len(wordList)}")
    print(wordList)


except FileNotFoundError as e:
    print(e)

