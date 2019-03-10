import requests
from bs4 import BeautifulSoup
import nltk
import os
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.util import ngrams
from nltk import ne_chunk
text=""
url = "https://en.wikipedia.org/wiki/Google"
source_code = requests.get(url)
plain_text = source_code.text
soup = BeautifulSoup(plain_text, "html.parser")
ptags= soup.find_all('p')
for i in ptags:
    text=text+i.getText()
infile=open('input.txt','w')
infile.write(text)
infile.close()
infile=open('input.txt','r')
lstemmer=LancasterStemmer()
lem=WordNetLemmatizer()
tokens=nltk.word_tokenize(infile.read())
for i in tokens:
    print(nltk.pos_tag(i))
    print(lstemmer.stem(i))
    print(lem.lemmatize(i))
tri= ngrams(tokens,3)
for j in tri:
    print(j)
print(ne_chunk(nltk.pos_tag(tokens)))