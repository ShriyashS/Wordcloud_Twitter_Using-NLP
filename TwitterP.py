
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:15:28 2019

@author: Shriyash Shende

"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
import re
from nltk.stem import WordNetLemmatizer



# Reading file
with open("RG.txt","r", encoding='utf8',errors='ignore') as rev:
    review = rev.read()
    
    
with open("sw.txt","r") as sw:
    stopwords = sw.read()
 
#Cleaning 
# Removing unwanted symbols incase if exists
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(review)
corpus = []

for i in sentences:
    review_1 = re.sub('[^a-zA-Z]', ' ',i)
    review_1= review_1.lower()
    review_1 = review_1.split()
    review_1 = [wordnet.lemmatize(word) for word in review_1 if not word in stopwords]
    review_1 = ' '.join(review_1)
    corpus.append(review_1)

ip_corpus =  " ".join(corpus) 
with open("ip_corpus_words.txt","w") as ip:
    ip.write(ip_corpus)


wordcloud_ip = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(ip_corpus)

plt.imshow(wordcloud_ip)


# positive words # Choose the path for +ve words stored in system
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]
positive_words_review = []
for i in sentences:
    review_1 = re.sub('[^a-zA-Z]', ' ',i)
    review_1= review_1.lower()
    review_1 = review_1.split()
    review_1 = [wordnet.lemmatize(word) for word in review_1 if  word in poswords]
    review_1 = ' '.join(review_1)
    positive_words_review.append(review_1)
    
ip_in_pos = " ".join(positive_words_review ) 
with open("ip_pos_words.txt","w") as ip:
    ip.write(ip_in_pos)

wordcloud_ip1 = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(ip_in_pos)

plt.imshow(wordcloud_ip1)
    
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")
  
  
negwords = negwords[37:]
negative_words_review = []
for i in sentences:
    review_1 = re.sub('[^a-zA-Z]', ' ',i)
    review_1= review_1.lower()
    review_1 = review_1.split()
    review_1 = [wordnet.lemmatize(word) for word in review_1 if  word in negwords]
    review_1 = ' '.join(review_1)
    negative_words_review.append(review_1)
    
ip_in_neg = " ".join(negative_words_review)
with open("ip_neg_words.txt","w") as ip:
    ip.write(ip_in_neg)

wordcloud_ip2 = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(ip_in_neg)

plt.imshow(wordcloud_ip2)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(min_df = 1, stop_words='english')
X = cv.fit_transform(corpus).toarray()
cv.get_feature_names()


