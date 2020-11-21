#wordnet
#wordnet is basically take the word and give us their meanings synonyms and antonyms as well

from nltk.corpus import wordnet

sysns = wordnet.synsets("program")

# #synset
# print(sysns)

# #just the word
# print(sysns[0].lemmas()[0].name())

# #definition
# print(sysns[0].definition())

# #examples
# print(sysns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas(): #lemmas both contains synonyms and antonyms
        # print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms() [0].name())
# print(set(synonyms))
# print(set(antonyms))



w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2)) #give similarity percentage btwen two words

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2)) #give similarity percentage btwen two words

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2)) #give similarity percentage btwen two words


