#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Implementation of Sentiment Analysis of airline tweets  

import numpy as np 
import pandas as pd 
import re
import nltk 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


## Got the below github link from online 
reviews_df  = pd.read_csv('Tweets.csv')

reviews_df = reviews_df['text']


# In[33]:


reviews_df = pd.DataFrame(reviews_df,columns=['text'])


# In[34]:


reviews_df.shape


# In[35]:


reviews_df.head()


# In[36]:


# return the wordnet object value corresponding to the POS tag
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADJ
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)

# clean text data
#reviews_df["review_clean"] = reviews_df["text"].apply(lambda x: clean_text(x))


# In[37]:


reviews_df = reviews_df[['text']]
reviews_df["review_clean"] = reviews_df["text"].apply(lambda x: clean_text(x))


# In[38]:


reviews_df.head()


# In[10]:


nltk.download('vader_lexicon')
# add sentiment anaylsis columns
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
reviews_df["sentiments"] = reviews_df["review_clean"].apply(lambda x: sid.polarity_scores(x))
reviews_df = pd.concat([reviews_df.drop(['sentiments'], axis=1), reviews_df['sentiments'].apply(pd.Series)], axis=1)


# In[11]:


## From the below data we come to know the neg, pos and neu point of a particular tweet
reviews_df.head()


# In[12]:


# add number of characters column
reviews_df["nb_chars"] = reviews_df["text"].apply(lambda x: len(x))

# add number of words column
reviews_df["nb_words"] = reviews_df["text"].apply(lambda x: len(x.split(" ")))


# In[14]:


reviews_df.head()


# In[15]:


# highest positive sentiment reviews (with more than 5 words)
reviews_df[reviews_df["nb_words"] >= 5].sort_values("pos", ascending = False)[["text", "pos"]].head(10)


# In[16]:


# lowest negative sentiment reviews (with more than 5 words)
reviews_df[reviews_df["nb_words"] >= 5].sort_values("neg", ascending = False)[["text", "neg"]].head(10)


# In[17]:


# wordcloud function

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color = 'white',
        max_words = 200,
        max_font_size = 40, 
        scale = 3,
        random_state = 42
    ).generate(str(data))

    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(reviews_df["text"])


# In[18]:


# create doc2vec vector columns
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from gensim.models import Word2Vec

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(reviews_df["review_clean"].apply(lambda x: x.split(" ")))]

# train a Doc2Vec model with our text data
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# transform each document into a vector data
doc2vec_df = reviews_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
reviews_df = pd.concat([reviews_df, doc2vec_df], axis=1)


# In[19]:


## Below is the most similar words for the 'Good' in the docs
model.most_similar('good')


# In[20]:


reviews_df.shape


# In[21]:


reviews_df.head()


# In[22]:


#The lower the IDF value of a word, the less unique it is to any particular document.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

##Initialize CountVectorizer

cv=CountVectorizer()
# this steps generates word counts for the words in your docs
word_count_vector=cv.fit_transform(reviews_df["review_clean"])

##Compute the IDF values
#Now we are going to compute the IDF values by calling tfidf_transformer.fit(word_count_vector) 

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

# print idf values
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
df_idf.sort_values(by=['idf_weights']).tail()


# In[23]:


## Compute the TFIDF score for your documents without Sklearn
# count matrix
count_vector=cv.transform(reviews_df["review_clean"])
 
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)

feature_names = cv.get_feature_names()
 
#get tfidf vector for first document or the first review 
first_document_vector=tf_idf_vector[0] ## 0 means first review
 
#print the scores
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False).head()


# In[24]:


count_vector


# In[24]:


# tf-idfs using sklearn and added to our dataset after appending it with Word_  
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_result = tfidf.fit_transform(reviews_df["review_clean"]).toarray()
tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
tfidf_df.index = reviews_df.index
reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)


# In[25]:


reviews_df.head()


# In[26]:


## Function to fetch the Noun words only
def extract_NN(sent):
    grammar = r"""
    NBAR:
        # Nouns and Adjectives, terminated with Nouns
        {<NN.*>*<NN.*>}

    NP:
        {<NBAR>}
        # Above, connected with in/of/etc...
        {<NBAR><IN><NBAR>}
    """
    chunker = nltk.RegexpParser(grammar)
    ne = set()
    chunk = chunker.parse(nltk.pos_tag(nltk.word_tokenize(sent)))
    for tree in chunk.subtrees(filter=lambda t: t.label() == 'NP'):
        ne.add(' '.join([child[0] for child in tree.leaves()]))
    return ne


# In[27]:


reviews_df['noun']= reviews_df['text'].apply(lambda x:extract_NN(x))


# In[28]:


reviews_df.head()


# So using the above sentiment we can distinguish the Positive Negative and Neutral review of the tweet, Also we have implemented the doc2vec and fetch the noun keywords from the tweet as well here.

# In[ ]:




