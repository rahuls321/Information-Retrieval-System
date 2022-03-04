from enum import unique
import os

import numpy as np
import nltk
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

from q1 import *
from q2 import *


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--corpus",required=True, help="path to input directory of english corpus")
    ap.add_argument("-vof", "--vocab_flag", type=int, default=0, help="Do you want to build vocab from scratch for each doc in the courpus \
                                                                or you want to use existed vocab available in the utils folder")
    ap.add_argument("-v", "--vocab_path", required=True, help="path to the existed vocab built on same corpus")                                                        
    ap.add_argument("-t", "--top_k", type=int, default=5,required=True, help="Top k documents will be retrieved")
    ap.add_argument("-pf", "--postings_flag", type=int, default=0, help="Do you want to build postings list from scratch for each doc in the corpus \
                                                                or you want to use existed postings list available in the utils folder")
    ap.add_argument("-p", "--postings_path",required=True, help="path to the existed postings list built on same corpus")                                                               
    ap.add_argument("-q", "--load_query", required=True, help="path to query files")
    ap.add_argument("-vef", "--vector_flag", type=int, default=0, help="Do you want to build tf-idf doc vector from scratch for each doc in the courpus \
                                                                or you want to use existed tf-idf doc vector available in the utils folder")
    ap.add_argument("-dv", "--doc_vector_path",required=True, help="path to the existed tf-idf doc vector built on same corpus")  
    ap.add_argument("-o", "--out_folder", required=True, help="path to output folder")

    args = vars(ap.parse_args())

    english_corpus_path = args["corpus"]
    top_k=args["top_k"]
    vocab_stem_path = args["vocab_path"]
    vocab_flag = args["vocab_flag"]
    postings_list_path = args["postings_path"]
    postings_flag = args["postings_flag"]
    query_file = args["load_query"]
    doc_vector_path = args["doc_vector_path"]
    vector_flag = args["vector_flag"]
    out_folder=args["out_folder"]

    if vocab_flag:
        print("Text cleaning and Text processing begins ...")
        vocab_doc_wise_stemming = create_vocab_from_corpus(english_corpus_path)
    else:
        # Loading Vocab
        print("Loading Vocab ...")
        vocab_doc_wise_stemming = np.load(vocab_stem_path, allow_pickle='TRUE').item()
    print("Total Documents in the corpous: ", len(vocab_doc_wise_stemming))

    #Choosing Snowball stemmer (advanced version of porter_stemmer)
    stemmer = get_stemmer('snowball_stemmer')

    doc_index = {}
    doc_lengths={}
    ind=0
    print("Preparing doc lengths dict and doc indexes dict ...")
    for doc_id, vocab in vocab_doc_wise_stemming.items():
        doc_index[ind] = doc_id
        doc_lengths[ind] = len(vocab)
        ind+=1

    if postings_flag:
        print("Building postings list ...")
        postings_list = create_postings_list(vocab_doc_wise_stemming)
    else:
        print("Extracting postings list ...")
        file = open(postings_list_path,'rb')
        postings_list = pickle.load(file)
        file.close()

    print("Total Vocab in the corpous: ", len(postings_list))

    #To visualize the postings list, Printing 5 instances from this list
    print("Postings list format: token -> [docID -> tokenFreq] -> [docID -> tokenFreq] ...")
    print("First 5 instances from doc and first 5 docID and freq.")
    p_idx=0
    for word, node in postings_list.items():
        print(word, end='->')
        p_j_idx=0
        while node is not None:
            print('[', end='')
            print(node.docID, end='->')
            print(node.freq, end=' ]->')
            node=node.next
            p_j_idx+=1
            if p_j_idx >= 5: break
        print('\n')
        p_idx+=1
        if(p_idx>=5): break


    ################################################################# Boolean IR ########################################################

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

    print("Query processing for Boolean, TF-IDF, BM25 IR system ...")
    queries = get_queries_from_file(query_file)

    queries_list = queries
    end_time=0
    score=0
    total_mapped_doc=[]
    print("Top {} documents will be retrieved".format(top_k))
    queries_not_matched=[]
    for query_idx, query in queries_list.items():
        st_time = time.time()
        print("Query ID: ", query_idx)
        query_tokens = query_preprocessing(query)
        matched_doc, token_not_found = find_matched_doc(query_tokens, postings_list, doc_index)
        total_mapped_doc.append(matched_doc[:top_k])
        # original_relevant_doc = [docID for docID, relevance in zip(queries_o_answer[query_idx], queries_relevant_doc[query_idx]) if relevance==1]
        # print("Ground truth doc: ", original_relevant_doc)
        # q_score, true_positive_doc =get_precision_score(matched_doc[:top_k], original_relevant_doc)
        # print("True positive doc: ", true_positive_doc)
        # score+=q_score
        # print("Precision score: ", q_score)
        # print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
        # if(q_score==0): queries_not_matched.append(query_idx)
        print("Top {} Matched Doc in ranked order: {}".format(top_k, matched_doc[:top_k]))
        print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
        end_time+=time.time()-st_time
        #print("These Tokens not found in the corpus: ", token_not_found)
    # pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'boolean_output.csv', header=False)
    create_output_file('booleanIR', total_mapped_doc, queries_list, top_k, out_folder)
    # print("Average Precision score: {}".format(score/len(queries_list)))
    # print("Queries not matched with any doc in corpus: ", queries_not_matched)
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


    # To prepare champion lists, uncomment below code
    # champion_lists = get_champion_lists(vocab_doc_wise_stemming, postings_list, doc_index)
    # champion_lists

    print("######################################## Tf-Idf Retrieval System started #################################################")

    if vector_flag:
        # To Prepare the doc vectors for all doc in the corpus 
        print("Preparing doc vectors ...")
        stt_time = time.time()
        doc_vectors = get_doc_tf_idf(postings_list, len(doc_index))
        print("Time taken in building TF-IDF doc vectors: {}".format(time.time() - stt_time))
    else:
        print("Extracting TF-IDF doc vectors ...")
        stt_time = time.time()
        # sparse_matrix = scipy.sparse.csc_matrix(doc_vectors)
        # scipy.sparse.save_npz('sparse_matrix_doc_vectors.npz', sparse_matrix)
        doc_vectors = scipy.sparse.load_npz(doc_vector_path).toarray()
        print("Time taken in extracting TF-IDF doc vectors: {}".format(time.time() - stt_time))
    print("TF-IDF Doc vector shape: ", doc_vectors.shape)

    queries_list = queries
    top_r=15
    end_time=0
    queries_not_matched=[]
    total_mapped_doc=[]
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
        total_mapped_doc.append(mapped_doc[:top_k])
        ##top_r_mapped_doc, token_not_found = top_R_ranked_doc(new_q_tokens, champion_lists, top_r)

        # original_relevant_doc = [docID for docID, relevance in zip(queries_o_answer[query_idx], queries_relevant_doc[query_idx]) if relevance==1]
        # print("Ground truth doc: ", original_relevant_doc)
        # q_score, true_positive_doc =get_precision_score(mapped_doc[:top_k], original_relevant_doc)
        # print("True positive doc: ", true_positive_doc)
        # score+=q_score
        # print("Precision score: ", q_score)
        # print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
        # if(q_score==0): queries_not_matched.append(query_idx)
        print("Top {} Matched Doc in ranked order: {}".format(top_k, mapped_doc[:top_k]))
        print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
        end_time+=time.time()-st_time
    # pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'tf_idf_output.csv', header=False)
    create_output_file('tf_idf_IR', total_mapped_doc, queries_list, top_k, out_folder)
    # print("Average Precision score: {}".format(score/len(queries_list)))
    # print("Queries not matched with any doc in corpus: ", queries_not_matched)
    print("Avg time takes to run TF-IDF Retrieval system for one query: {:.5f} sec".format(end_time/len(queries_list)))


    ############################################################## BM25 ########################################################

    print("######################################## BM25 IR system started #################################################")

    # ## BM25

    # These are the steps considered
    # 1. Tokenize the query into tokens and remove the stop words and also remove if there's any non-ascii characters
    # 2. Get local weight by modified term frequency formula $$\frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$
    # 3. Get global weight by inverse doc frequency as the priors aren't given by given formula $$\log \frac{n}{df_t}$$
    # 4. Get RSVd score using below formula and based on this score, select top k documents $$RSVd = \sum_{\forall t \in q} \left(\log \frac{n}{df_t}\right) . \frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$

    queries_list = queries
    end_time=0
    print("Top {} documents retrieved".format(top_k))
    k1 = np.random.uniform(1.2, 2.0)
    b=0.75
    score=0
    queries_not_matched=[]
    total_mapped_doc = []
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
        total_mapped_doc.append(mapped_doc[:top_k])
        # original_relevant_doc = [docID for docID, relevance in zip(queries_o_answer[query_idx], queries_relevant_doc[query_idx]) if relevance==1]
        # print("Ground truth doc: ", original_relevant_doc)
        # q_score, true_positive_doc =get_precision_score(mapped_doc[:top_k], original_relevant_doc)
        # score+=q_score
        # print("Precision score: ", q_score)
        # print("Out of all {} relelvant doc: {} relevant documents are retrieved".format(len(original_relevant_doc), len(true_positive_doc)))
        # if(q_score==0): queries_not_matched.append(query_idx)
        print("Top {} Matched Doc in ranked order: {}".format(top_k, mapped_doc[:top_k]))
        print("Time taken to process Query id: {}: {:.5f} sec".format(query_idx, time.time()-st_time))
        end_time+=time.time()-st_time
        #print("These Tokens not found in the corpus: ", token_not_found)
    # pd.DataFrame.from_dict(data=qid_mapped_doc_dict, orient='index').to_csv(out_folder+'bm25_output.csv', header=False)
    create_output_file('bm25_IR', total_mapped_doc, queries_list, top_k, out_folder)
    # print("Average Precision score: {}".format(score/len(queries_list)))
    # print("Queries not matched with any doc in corpus: ", queries_not_matched)
    print("Avg time takes to run BM25 Retrieval system for one query: {:.5f} sec".format(end_time/len(queries_list)))

