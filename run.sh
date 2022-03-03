#!/bin/bash

top_k=5
corpus="./data/english-corpora/"
vocab_path='./utils/vocab_doc_wise_stemming.npy'
postings_path="./utils/postings_list.pkl"
query_file='./data/query.txt'
query_ground_truth='./data/query_answers.txt'
doc_vector_path='./utils/sparse_matrix_doc_vectors.npz'
out_folder='./output/'
vocab_flag=0
postings_flag=0
vector_flag=0

python main.py -c $corpus --vocab_flag $vocab_flag -v $vocab_path -t $top_k --postings_flag $postings_flag -p $postings_path -q $query_file -g $query_ground_truth --vector_flag $vector_flag -dv $doc_vector_path -o $out_folder