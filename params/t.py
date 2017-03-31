import codecs

f = codecs.open("wo.txt","w","utf8")
for line in open("word_dict.txt"):
    f.write(line.split()[-1].decode("utf-8")+'\n')
f.close()
