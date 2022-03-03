#!/usr/bin/env python
# coding: utf-8

# In[47]:


import re
import os
import nltk
import unidecode
import numpy as np
import codecs


# In[2]:


# ## Text Cleaning
# 1. Text is splitted by '\t' first.
# 2. Remove extra spaces
# 3. Tokenize the strings
# 4. Remove Punctuations from tokenize words
# 5. Remove Number
# 6. Remove Double quotations from tokens
# 7. Replace URL with url tag
# 8. Remove ascents from string using decode like A&deg;
# 9. Split camelCase word into 'camel' and 'Case'
# 10. Remove number from token

# In[3]:


def isExtraSpace(text):
    return len(text)==0


# In[4]:


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens


# In[5]:


def isASCII(token):
    if token in ['.','+','*','?','[','/', '//','\\','^','%',']', '$','(',')','{','}','=', '!', '|',':','-']:
        return True
    return False


# In[6]:


def ignore_special_token(token):
    if token in ['^^^^', '==']:
        return True
    return False


# In[7]:


def no_change(token):
    if token in ['Objective-J', 'I/O']:
        return True
    return False


# In[8]:


def is_number(token):
    num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
    return bool(num_regex.match(token))


# In[9]:


def replace_url(token):
    replaced_text = re.sub('(http[s]?://)?((www)\.)?([a-zA-Z0-9]+)\.{1}((com)(\.(cn))?|(org))', '<url>', token)
    return replaced_text


# In[10]:


def split_camelCase(token):
    return re.findall(r'[a-zA-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', token)


# In[11]:


def remove_numbers_from_token(token):
    return re.sub(r'[0-9]+', '', token)


# In[41]:


def get_stemmer(stemmer_type):
    if(stemmer_type=='porter_stemmer'): stemmer = nltk.PorterStemmer()
    elif(stemmer_type=='snowball_stemmer'): stemmer = nltk.SnowballStemmer(language = 'english')
    return stemmer


# In[34]:


def text_cleaning(tokens):
    new_tokens=[]
    for token in tokens:
        token=token.replace('"', '')
        token = replace_url(token)
        token = unidecode.unidecode(token)
        if isASCII(token) or is_number(token) or len(token)==0: continue
        if no_change(token):
            new_tokens.append(token)
            continue
        tt = re.split(', |:|\+|%|\|/|\*|$|&|@|_|-|!|;|,', token)
        for t in tt:
            t=t.replace('"', '')
            t=unidecode.unidecode(t)
            #Removing ASCII character and extra space after splitting
            if isASCII(t) or len(t)<=1 or is_number(t): continue
            #Removing '.' from the word
            split_by_dot = t.split('.')
            final_token = sorted(split_by_dot, key=len, reverse=True)[0]
            #If still some non-ascii char left in the token 
            if not final_token.isalnum():
                final_token = re.sub('[^A-Za-z0-9]+', '', final_token)
            #Remove number from token
            final_token = remove_numbers_from_token(final_token)
            #CamelCase condition based on first char if it is lower or not
            if final_token and final_token[0].islower():
                camel_tokens = split_camelCase(final_token)
                for tt in camel_tokens: 
                    if len(tt) <= 1: continue
                    new_tokens.append(tt.lower())
            else: 
                if len(final_token) <= 1: continue
                new_tokens.append(final_token.lower())
    return new_tokens


def create_vocab_from_corpus(file_path):
    vocab_doc_wise_tokenization={}
    vocab_doc_wise_stemming={}
    i=0
    for file in os.listdir(file_path):
        print(i, end=' ')
        print(file)
        with codecs.open(os.path.join(file_path, file), mode='r', encoding='utf-8') as input_file:
            next(input_file)
            d=[]
            for line in input_file:
                text = line.strip().split('\t')[0]
                if(isExtraSpace(text)): continue
                text=text.replace('"', '')
                ####################################### TOKENIZATION ################################################
                tokens = tokenize(text)
                pun_free_token = text_cleaning(tokens)
                if pun_free_token:
    #                 print("Final: ", pun_free_token)
                    d = sum([], d+pun_free_token)
        ##d = sorted(set(d), key=lambda x:d.index(x))
        doc_name = os.path.splitext(file)[0]
        vocab_doc_wise_tokenization[doc_name] = d
        ####################################### STEMMING ################################################
        stemmer = get_stemmer('snowball_stemmer')
        st = [stemmer.stem(word) for word in d]
        vocab_doc_wise_stemming[doc_name] = st
        i+=1
        # if i>=10: break
    # np.save('vocab_doc_wise_tokenization.npy', vocab_doc_wise_tokenization) 
    # np.save('vocab_doc_wise_stemming.npy', vocab_doc_wise_stemming) 

    # vocab_doc_wise_tokenization = np.load('vocab_doc_wise_tokenization.npy', allow_pickle='TRUE').item()
    # vocab_doc_wise_stemming = np.load('vocab_doc_wise_stemming.npy', allow_pickle='TRUE').item()
    ##print(vocab_doc_wise_tokenization)
    #print(vocab_doc_wise_stemming)
    return vocab_doc_wise_stemming





