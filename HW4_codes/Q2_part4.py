dict_e10 = freq_count("./e10.txt")
print(dict_e10)
bag_of_words = []
for k,v in dict_e10.items():
  bag_of_words.append(v)
print(bag_of_words)