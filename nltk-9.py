#corpora
#corpus is a library where so many texts are present and we can tokenize it

from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sampleText = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sampleText)
print(tok[5:15])


