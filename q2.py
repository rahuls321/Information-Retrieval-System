#!/usr/bin/env python
# coding: utf-8

# In[46]:
from enum import unique
import os
# os.environ["NLTK_DATA"] = "/data/rahulk/nltk_data"

import numpy as np
import nltk
# nltk.data.path.append('/data/rahulk/nltk_data')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import math
import time
import pickle
import codecs
import sys
import scipy.sparse
import pandas as pd
import argparse
sys.setrecursionlimit(80000)


def get_stemmer(stemmer_type):
    if(stemmer_type=='porter_stemmer'): stemmer = nltk.PorterStemmer()
    elif(stemmer_type=='snowball_stemmer'): stemmer = nltk.SnowballStemmer(language = 'english')
    return stemmer

def create_output_file(system_type, total_mapped_doc, queries_list, top_k, out_folder):
    total_mapped_doc=np.array(total_mapped_doc).flatten()
    total_mapped_doc=total_mapped_doc.reshape(len(total_mapped_doc), 1)
    query_ids = [qid for qid in list(queries_list.keys()) for _ in range(top_k)]
    query_ids = np.array(query_ids).reshape(len(total_mapped_doc), 1)
    ones = np.ones((len(total_mapped_doc), 1), dtype=int)
    save_query_file = np.hstack((query_ids,ones,total_mapped_doc,ones))
    # print(save_query_file)

    with open(out_folder+system_type+'.txt', 'w') as f:
        for x in save_query_file:
            f.write('%s,%s,%s,%s\n' % tuple(x))



#Creating Node which has three sub-nodes containing document ID, freq of word in that docID 
#and next to link with next docID
class Node:
    def __init__(self, docID, freq=None):
        self.docID = docID
        self.freq = freq
        self.next = None

#Creating word freq for each doc
def get_word_freq(vocab):
    word_freq={}
    for word in vocab:
        if word in word_freq.keys():
            word_freq[word]+=1
        else: word_freq[word]=1
    return word_freq


# ### Creating Postings list
## Postings lists are basically dictionary where key contains vocab and values contain linked list node 
# having two subnode one contains docID and other contains freq of vocab in the given docID
def create_postings_list(vocab_doc_wise_stemming):
    postings_list = {}
    doc_index = {}
    ind=0
    doc_lengths={}
    for doc_id, vocab in vocab_doc_wise_stemming.items():
        word_freq = get_word_freq(vocab)
        for word, freq in word_freq.items():
            if word in postings_list.keys():
                firstNode = postings_list[word]
                while firstNode.next is not None:
                    firstNode = firstNode.next
                firstNode.next = Node(ind, freq)
            else:
                postings_list[word] = Node(ind, freq)
        doc_index[ind] = doc_id
        doc_lengths[ind] = len(vocab)
        ind+=1
        if ind>=10: break

    # filehandler = open(postings_list_path,"wb")
    # pickle.dump(postings_list,filehandler)

    return postings_list



#This function check nonASCII character while preprocessing query tokens
def isnonASCII(token):
    if token in ['.','+','*','?','[','/', '//','\\','^','%',']', '$','(',')','{','}','=', '!', '|',':','-', ',', ';']:
        return True
    return False

def is_operator(token):
    if token in ['&' , '|']:
        return True
    return False

#Precedence of operators
def precedence_oper(token):
    if token=='&': return 2
    elif token=='|': return 1
    else: return -1

def get_postfix_list(tokens):
    stack = []
    postfix_list = []
    for token in tokens:
        #If token is left small bracket '('
        if token == '(': stack.append(token)
        elif token == ')':
            while(len(stack)>0 and stack[-1]!='('):
                postfix_list.append(stack.pop())
            if len(stack)==0 and token==')':
                raise ValueError('Either unnecessary parenthesis or Not a balanced query')
            stack.pop()
            if len(stack)>0 and stack[-1] == '(':
                raise ValueError('Either unnecessary parenthesis or Not a balanced query')
        elif is_operator(token):
            while(len(stack)>0 and precedence_oper(token) <= precedence_oper(stack[-1])):
                postfix_list.append(stack.pop())
            stack.append(token)
        else: 
            postfix_list.append(token)
    while len(stack)>0:
        postfix_list.append(stack.pop())
    return postfix_list


def query_preprocessing(q):
    #Remove stop words from query
    stop_words = set(stopwords.words('english'))
    #Tokenize query first
    q_tokens = word_tokenize(q)
    updated_q_tokens=[]
    connecting_words = {'and':'&','AND':'&', 'or':'|','OR':'|', 'not':'~','NOT':'~'}
    for t, token in enumerate(q_tokens):
        if token in list(connecting_words.keys()):
            if token=='not' or token=='NOT':
                if t+1>=len(q_tokens):
                    #raise ValueError("Invalid query!")
                    print("Invalid Query!!")
                    continue
                else:
                    updated_q_tokens.append('~'+q_tokens[t+1])
                    q_tokens.remove(q_tokens[t+1])
            else: updated_q_tokens.append(connecting_words[token])
        else:
            updated_q_tokens.append(token)
    #print(updated_q_tokens)
    stemmer = get_stemmer('snowball_stemmer')
    new_q_tokens = [stemmer.stem(word.lower()) for word in updated_q_tokens if word not in stop_words and not isnonASCII(word) and len(word)>1]
    #print(new_q_tokens)
    #Convert this infix list into postfix list to process operator in right way
    q_tokens = get_postfix_list(new_q_tokens)
    return q_tokens


def get_binary_vec(token, postings_list, doc_size):
    word_embedd = np.zeros(doc_size, dtype=int)
    vocab = postings_list.keys()
    negation = False
    token_not_found=0
    if token[0]=='~':
        negation=True
        token=token[1:]
    if token not in vocab:
        #print("'"+token + "' was not found in the corpus")
        token_not_found=1
        return word_embedd, token_not_found
    node = postings_list[token]
    while node is not None:
        word_embedd[node.docID] = 1
        node=node.next
    if negation:
        word_embedd = np.invert(word_embedd)
    return word_embedd, token_not_found


def find_matched_doc(query_tokens, postings_list, doc_index):
    word_embedd_stack = []
    doc_size = len(doc_index)
    token_not_found=[]
    stemmer = get_stemmer('snowball_stemmer')
    for token in query_tokens:
        if is_operator(token):
            if(len(word_embedd_stack)<2): 
                print("Invalid Query!!")
                raise ValueError("Query is not correct or use more stopping words")
            first_operand = word_embedd_stack.pop()
            second_operand = word_embedd_stack.pop()
            
            if token=='&': word_embedd_stack.append(first_operand & second_operand)
            elif token=='|': word_embedd_stack.append(first_operand | second_operand)
            else:
                raise ValueError('Can\'t process this operator: ', token)
        else:
            st = stemmer.stem(token)
            
            token_embedd, flag = get_binary_vec(token, postings_list, doc_size)
            if(flag): token_not_found.append(token)
            word_embedd_stack.append(token_embedd)
    matched_doc = [doc_index[docID] for docID in np.where(word_embedd_stack[-1])[0]]
    return matched_doc, token_not_found


def get_precision_score(mapped_doc, ground_truth):
    intersection = set(mapped_doc).intersection(set(ground_truth))
    if len(mapped_doc) == 0:
        return 0, intersection
    return len(intersection)/len(mapped_doc), intersection

def get_queries_from_file(query_file, relevant_doc_path):

    with open(query_file, mode='r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        queries={}
        for line in lines:
            query = line.strip().split('\t')
            queries[query[0]] = query[1]
    # print(b_queries)
    with open(relevant_doc_path, mode='r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        queries_o_answer={}
        queries_relevant_doc={}
        for line in lines:
            query = line.strip().split(',')
            if len(query) != 4:
                print("Not in QRels format.")  
                break 
            assert len(query) == 4
            if query[0] in queries_o_answer.keys():
                queries_o_answer[query[0]].append(query[2])
                queries_relevant_doc[query[0]].append(int(query[3]))
            else: 
                queries_o_answer[query[0]] = [query[2]]
                queries_relevant_doc[query[0]] = [int(query[3])]

    return queries, queries_o_answer, queries_relevant_doc





def get_doc_tf_idf(postings_list, doc_size):
    doc_vector = np.zeros((doc_size, len(postings_list.keys())), dtype=float)
    j=0
    for token, node in postings_list.items():
        print("DocID: ", j)
        node = postings_list[token]
        doc_IDS=[]
        dft=0
        idx = list(postings_list.keys()).index(token)
        while node is not None:
            doc_vector[node.docID][idx] = node.freq
            doc_IDS.append(node.docID)
            node=node.next
            dft+=1
        for doc_ID in doc_IDS:
            doc_vector[doc_ID][idx] *= math.log(doc_size / dft)
        j+=1

        if j>=10: break
            
    for idx, d_vec in enumerate(doc_vector):
        d_vec_len = np.linalg.norm(d_vec)
        if d_vec_len != 0:
            doc_vector[idx] /= d_vec_len
        
    return doc_vector

def get_query_vector(query_tokens, postings_list):
    query_vector = np.zeros(len(postings_list.keys()), dtype=int)
    for token in query_tokens:
        if token not in postings_list.keys():
            #print("'"+token + "' was not found in corpus")
            continue
        else:
            idx = list(postings_list.keys()).index(token)
            query_vector[idx] += 1
    return query_vector


#To create champion lists
def get_champion_lists(vocab_doc_wise_stemming, postings_list, doc_index):
    champion_lists={token:[] for token in list(postings_list.keys())}
    doc_name=[]
    full_doc_tf_idf=[]
    for doc, doc_vocab in vocab_doc_wise_stemming.items():
        idx = list(doc_index.values()).index(doc)
        doc_tf_idf = get_doc_tf_idf(doc_vocab, postings_list, idx, len(doc_index))
        full_doc_tf_idf.append(doc_tf_idf)
        doc_name.append(doc)
    full_doc_tf_idf = np.array(full_doc_tf_idf)
    token_idx=0
    for token in list(champion_lists.keys()):
        get_doc_cos_sim_vec = [(score, d_name) for score, d_name in sorted(zip(full_doc_tf_idf[:, token_idx], doc_name), reverse=True)]
        champion_lists[token] = get_doc_cos_sim_vec
        token_idx+=1
    return champion_lists

def top_R_ranked_doc(query_tokens, champion_lists, top_r):
    top_R_doc_vec = []
    token_not_found=[]
    for token in query_tokens:
        if token not in champion_lists.keys():
            token_not_found.append(token)
        else:
            doc_vec = champion_lists[token][:top_r]
            top_R_doc_vec = sum([], top_R_doc_vec+[score_doc_tuple for score_doc_tuple in doc_vec])
    top_R_doc_vec = sorted(top_R_doc_vec, key=lambda x: x[0], reverse=True)
    mapped_doc=[]
    for score_vec_t in top_R_doc_vec:
        if score_vec_t[1] not in mapped_doc:
            mapped_doc.append(score_vec_t[1])
    return mapped_doc, token_not_found


def get_sim_score(V_q, doc_vectors, doc_index):
    q_sim_score=[]
    q_sim_score_map_doc_name=[]
    for doc_ID, doc_vec in enumerate(doc_vectors):
        v_d = doc_vec
        cos_sim =  np.dot(V_q, v_d)
        q_sim_score.append(cos_sim)
        q_sim_score_map_doc_name.append(doc_index[doc_ID])
    ranked_doc = [doc_name for score, doc_name in sorted(zip(q_sim_score, q_sim_score_map_doc_name), reverse=True)]
    return ranked_doc


def get_BM25(query_tokens, postings_list, doc_lengths, k1=1.2, b=0.75):
    #Each dimension corresponding to one document
    RSVd_vec = np.zeros(len(doc_lengths), dtype=float)
    token_not_found=[]
    for token in query_tokens:
        dft=0
        doc_vector = np.zeros(len(doc_lengths), dtype=int)
        if (token not in postings_list.keys()):
            #print("'"+token + "' was not found in corpus")
            token_not_found.append(token)
        else:
            node = postings_list[token]
            while node is not None:
                dft+=1
                doc_vector[node.docID]=node.freq
                node=node.next
        doc_vector = np.array(doc_vector, dtype=float)
        ## print(doc_vector)
        if dft==0: 
            RSVd_vec += doc_vector
        else:
            #Get Local weight
            L_d = np.array(list(doc_lengths.values()), dtype=float)
            L_avg = np.mean(list(doc_lengths.values()))
            local_wt_num = (k1+1)*doc_vector
            local_wt_den = k1*(1 - b + (b/L_avg)*L_d) + doc_vector
            local_wt = np.divide(local_wt_num, local_wt_den)
            #Get Global weight
            doc_size = len(doc_lengths)
            global_wt = math.log(doc_size / dft)
            ##print("global_wt: ", global_wt)
        #print("RSVd: ", RSVd_vec)
        #Multiply local weigth with global weight
            RSVd_vec += local_wt*global_wt
    return RSVd_vec, token_not_found
