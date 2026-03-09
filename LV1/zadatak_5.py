ham_word_count = 0
ham_msg_count = 0
spam_word_count = 0
spam_msg_count = 0
spam_with_exclamation = 0


try:
    fhand = open("LV1/textFiles/SMSSpamCollection.txt", encoding="utf-8")

    for line in fhand:
        line = line.rstrip()
    
        lineSplit = line.split(maxsplit = 1)

        label = lineSplit[0]
        message = lineSplit[1]
        words = message.split()
        num_words = len(words)

        if label == "ham":
            ham_word_count += num_words
            ham_msg_count += 1
        elif label == "spam":
            spam_word_count += num_words
            spam_msg_count += 1

            if message.endswith("!"):
                        spam_with_exclamation += 1

    fhand.close()

    if(ham_msg_count > 0):
         avg_ham = ham_word_count/ham_msg_count
    else:
         avg_ham = 0
    if(spam_msg_count > 0):
         avg_spam = spam_word_count/spam_msg_count
    else:
         avg_spam = 0

    print(f"a) Prosječan broj riječi u ham porukama: {avg_ham:.2f}")
    print(f"   Prosječan broj riječi u spam porukama: {avg_spam:.2f}")
    print(f"b) Broj spam poruka koje završavaju uskličnikom: {spam_with_exclamation}")

except FileNotFoundError as e:
    print(e)

