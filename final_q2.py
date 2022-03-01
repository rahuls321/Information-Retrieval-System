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
sys.setrecursionlimit(80000)


top_k=10
vocab_stem_path = './utils/vocab_doc_wise_stemming.npy'
postings_list_path = "./utils/postings_list.pkl"
query_file='./data/query.txt'
query_ground_truth='./data/query_answers.txt'
doc_vector_path = './utils/sparse_matrix_doc_vectors.npz'
out_folder='./output/'

# vocab_doc_wise_tokenization = np.load('vocab_doc_wise_tokenization.npy', allow_pickle='TRUE').item()
vocab_doc_wise_stemming = np.load(vocab_stem_path, allow_pickle='TRUE').item()
print("Total Documents in the corpous: ", len(vocab_doc_wise_stemming))

def get_stemmer(stemmer_type):
    if(stemmer_type=='porter_stemmer'): stemmer = nltk.PorterStemmer()
    elif(stemmer_type=='snowball_stemmer'): stemmer = nltk.SnowballStemmer(language = 'english')
    return stemmer

#Choosing Snowball stemmer (advanced version of porter_stemmer)
stemmer = get_stemmer('snowball_stemmer')

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

    filehandler = open(postings_list_path,"wb")
    pickle.dump(postings_list,filehandler)

    return postings_list

doc_index = {}
doc_lengths={}
ind=0
print("Preparing doc lengths dict and doc indexes dict ...")
for doc_id, vocab in vocab_doc_wise_stemming.items():
    doc_index[ind] = doc_id
    doc_lengths[ind] = len(vocab)
    ind+=1

print("Extracting postings list ...")
file = open(postings_list_path,'rb')
postings_list = pickle.load(file)
file.close()
print("Total Vocab in the corpous: ", len(postings_list))


##To visualize the postings list, please uncomment below code 
# for word, node in postings_list.items():
#     print(word, end='->')
#     while node is not None:
#         print(node.docID, end='->')
#         print(node.freq, end=' ')
#         node=node.next
#     print('\n')


# ### Query preprocessing

# Steps followed
# 1. Tokenize the query
# 2. Convert infix query expression to postfix query expression using stack approach
#         a. Check if the given expression is balanced or not
#         b. Check is there any extra parenthesis in the expression
# 3. Processing two operator only in the query **\&**(and) , **\|** (or) and **\~**(negation) and giving higeher precedence to the former
# 4. Using **snowball_stemmer** as a stemmer algorithm to find the stem word in the given query
# 5. Generate binary vector based on document size and consider negation sign as well while processing
# 6. Find document which contains the query word using **find_matched_doc** function and return a binary vector that shows which document contains that word
# 7. Remove stop words from query

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


print("Query processing for Boolean IR system ...")
b_queries, b_queries_o_answer, b_queries_relevant_doc = get_queries_from_file(query_file, query_ground_truth)

queries_list = b_queries
end_time=0
score=0
qid_mapped_doc_dict={}
print("Top {} documents will be retrieved".format(top_k))
queries_not_matched=[]
for query_idx, query in queries_list.items():
    st_time = time.time()
    print("Query ID: ", query_idx)
    query_tokens = query_preprocessing(query)
    matched_doc, token_not_found = find_matched_doc(query_tokens, postings_list, doc_index)
    qid_mapped_doc_dict[query_idx] = matched_doc[:top_k]
    original_relevant_doc = [docID for docID, relevance in zip(b_queries_o_answer[query_idx], b_queries_relevant_doc[query_idx]) if relevance==1]
    print("Ground truth doc: ", original_relevant_doc)
    q_score, true_positive_doc =get_precision_score(matched_doc[:top_k], original_relevant_doc)
    print("True positive doc: ", true_positive_doc)
    score+=q_score
    print("Precision score: ", q_score)
    print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
    if(q_score==0): queries_not_matched.append(query_idx)
    print("Total Matched Doc in ranked order: ", matched_doc[:top_k])
    print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
    end_time+=time.time()-st_time
    #print("These Tokens not found in the corpus: ", token_not_found)
pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'boolean_output.csv', header=False)
print("Average Precision score: {}".format(score/len(queries_list)))
print("Queries not matched with any doc in corpus: ", queries_not_matched)
print("Avg time takes to run Boolean Retrieval system for one query: {:.5f} sec".format(end_time/len(queries_list)))

########################################################################## TF-IDF ###########################################################################

# ## Tf-Idf Retrieval System

# Steps to consider
# 1. Tokenize the query first and remove the stopwords from the query and also remove nonASCII character
# 2. Find the query vector where each dimension represents freq. of token present in the query
# 3. For fast query processing, I have created one doc_vectors which is of shape (doc_size, total vocab) previously and saved their vectors in sparse_matrix_doc_vectors.npz
# 4. In doc vectors, rows represent all the documents and columns represent all the vocab present in all the documents 
# 5. Doc vectors are sparse matrix inorder to save space I used scipy.sparse.matrix to save all doc vectors.
# 6. Return top-k documents only.
# 7. I have also tried to prepare champion lists that is nothing but for each vocab in the documents find the rank of documents corresponding to them. But
#   it was taking too much space more than 24 GB) and also taking more than 24 hrs to prepare those lists. But I have provided the code for the same.
# 8. While calculating similarity score we don't need to normalize the query vector as even without normalization, product of V_q and v_d are much higher.


print("Query processing for TF-IDF and BM25 ...")
queries, queries_o_answer, queries_relevant_doc = get_queries_from_file(query_file, query_ground_truth)


def get_doc_tf_idf(postings_list, doc_size):
    doc_vector = np.zeros((doc_size, len(postings_list.keys())), dtype=float)
    j=0
    for token, node in postings_list.items():
        print(j)
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

# To prepare champion lists, uncomment below code
# champion_lists = get_champion_lists(vocab_doc_wise_stemming, postings_list, doc_index)
# champion_lists

print("######################################## Tf-Idf Retrieval System started #################################################")

print("Extracting TF-IDF doc vectors ...")
stt_time = time.time()
# To Prepare the doc vectors for all doc in the corpus 
# doc_vectors = get_doc_tf_idf(postings_list, len(doc_index))
# sparse_matrix = scipy.sparse.csc_matrix(doc_vectors)
# scipy.sparse.save_npz('sparse_matrix_doc_vectors.npz', sparse_matrix)
doc_vectors = scipy.sparse.load_npz(doc_vector_path).toarray()
print("TF-IDF Doc vector shape: ", doc_vectors.shape)
print("Time taken in extracting TF-IDF doc vectors: {}".format(time.time() - stt_time))


queries_list = queries
top_r=15
end_time=0
queries_not_matched=[]
qid_mapped_doc_dict={}
print("Top {} documents will be retrieved".format(top_k))
score=0
for query_idx, query in queries_list.items():
    st_time = time.time()
    print("Query ID: ", query_idx)
    #Tokenize query first
    q_tokens = word_tokenize(query)
    #Remove stop words from query
    stop_words = set(stopwords.words('english'))
    new_q_tokens = [stemmer.stem(word.lower()) for word in q_tokens if word not in stop_words and not isnonASCII(word)]
    V_q = get_query_vector(new_q_tokens, postings_list)
    mapped_doc = get_sim_score(V_q, doc_vectors, doc_index)
    qid_mapped_doc_dict[query_idx] = mapped_doc[:top_k]
    ##top_r_mapped_doc, token_not_found = top_R_ranked_doc(new_q_tokens, champion_lists, top_r)

    original_relevant_doc = [docID for docID, relevance in zip(queries_o_answer[query_idx], queries_relevant_doc[query_idx]) if relevance==1]
    print("Ground truth doc: ", original_relevant_doc)
    q_score, true_positive_doc =get_precision_score(mapped_doc[:top_k], original_relevant_doc)
    print("True positive doc: ", true_positive_doc)
    score+=q_score
    print("Precision score: ", q_score)
    print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
    if(q_score==0): queries_not_matched.append(query_idx)
    print("Total Matched Doc in ranked order: ", mapped_doc[:top_k])
    print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
    end_time+=time.time()-st_time
pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'tf_idf_output.csv', header=False)
print("Average Precision score: {}".format(score/len(queries_list)))
print("Queries not matched with any doc in corpus: ", queries_not_matched)
print("Avg time takes to run TF-IDF Retrieval system for one query: {:.5f} sec".format(end_time/len(queries_list)))


############################################################## BM25 ########################################################

print("######################################## BM25 IR system started #################################################")

# ## BM25

# These are the steps considered
# 1. Tokenize the query into tokens and remove the stop words and also remove if there's any non-ascii characters
# 2. Get local weight by modified term frequency formula $$\frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$
# 3. Get global weight by inverse doc frequency as the priors aren't given by given formula $$\log \frac{n}{df_t}$$
# 4. Get RSVd score using below formula and based on this score, select top k documents $$RSVd = \sum_{\forall t \in q} \left(\log \frac{n}{df_t}\right) . \frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$


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


queries_list = queries
end_time=0
print("Top {} documents retrieved".format(top_k))
k1 = np.random.uniform(1.2, 2.0)
b=0.75
score=0
queries_not_matched=[]
total_mapped_doc = []
qid_mapped_doc_dict={}
for query_idx, query in queries_list.items():
    st_time=time.time()
    print("Query ID: ", query_idx)
    #Tokenize query first
    q_tokens = word_tokenize(query)
    #Remove stop words from query
    stop_words = set(stopwords.words('english'))
    new_q_tokens = [stemmer.stem(word.lower()) for word in q_tokens if word not in stop_words and not isnonASCII(word)]
    #print(new_q_tokens)
    RSVd_vec, token_not_found = get_BM25(new_q_tokens, postings_list, doc_lengths, k1, b)
    mapped_doc = [docName for score, docName in sorted(zip(RSVd_vec, list(doc_index.values())), reverse=True)]
    qid_mapped_doc_dict[query_idx] = mapped_doc[:top_k]
    original_relevant_doc = [docID for docID, relevance in zip(queries_o_answer[query_idx], queries_relevant_doc[query_idx]) if relevance==1]
    print("Ground truth doc: ", original_relevant_doc)
    q_score, true_positive_doc =get_precision_score(mapped_doc[:top_k], original_relevant_doc)
    score+=q_score
    print("Precision score: ", q_score)
    print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
    if(q_score==0): queries_not_matched.append(query_idx)
    print("Total Matched Doc in ranked order: ", mapped_doc[:top_k])
    print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
    end_time+=time.time()-st_time
    #print("These Tokens not found in the corpus: ", token_not_found)
pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'bm25_output.csv', header=False)
print("Average Precision score: {}".format(score/len(queries_list)))
print("Queries not matched with any doc in corpus: ", queries_not_matched)
print("Avg time takes to run BM25 Retrieval system for one query: {:.5f} sec".format(end_time/len(queries_list)))

# total_mapped_doc=np.array(total_mapped_doc).flatten()
# total_mapped_doc=total_mapped_doc.reshape(len(total_mapped_doc), 1)
# query_ids = [qid for qid in list(queries_list.keys()) for _ in range(10)]
# query_ids = np.array(query_ids).reshape(len(total_mapped_doc), 1)
# ones = np.ones((len(total_mapped_doc), 1), dtype=int)
# save_query_file = np.hstack((query_ids,ones,total_mapped_doc,ones))
# print(save_query_file)

# with open('output.txt', 'w') as f:
#     for x in save_query_file:
#         f.write('%s,%s,%s,%s\n' % tuple(x))
