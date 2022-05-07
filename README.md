Assignment 1


*** NOTE *** 
1. Make sure you are connected with internet. (If you are running on cse server run authenticator.py file to bypass the firewall)
2. Make sure all three files downloaded in utils folder and total 3 file sizes will combinely 581 MB around otherwise download from this gdrive link - 
https://drive.google.com/drive/folders/1aT76iVKRdgBf6n9vYB6vatdkcCjySafF?usp=sharing 
3. Put english-corpora folder inside the data folder (Download from here - https://www.cse.iitk.ac.in/users/arnabb/ir/english/)



This folder contains following directories and files 
1. data - contains all the data used in this assignment 1
    a.  englis-corpora - english corpus folder contains all documents
    b.  query.txt - contains a set of 20 queries with proper format mentioned in assignment manual
    c.  query_ground_truth.txt - contains the list of 10 relevant doc in QRels format.
2. utils - It contains all necessary file which takes time to build like postings_list, vocab doc wise, doc vectors
    a.  vocab_doc_wise_stemming.npy - A dictionary where keys are docID and values are the vocab present in that documents.
    b.  postings_list.pkl - Postings lists are basically dictionary where key contains vocab and values contain linked list node having two subnode one contains docID and other contains freq of vocab in the given docID.
    c.  sparse_matrix_doc_vectors.npz - This is Doc vectors file required for TF-IDF system. For fast query processing, I have created one doc_vectors which is of shape (doc_size, total vocab) and saved their vectors in sparse_matrix_doc_vectors.npz. In doc vectors, rows represent all the documents and columns represent all the vocab present in all the documents. Doc vectors are sparse matrix inorder to save space I used scipy.sparse.matrix to save all doc vectors.
3. output - a folder contains all the generated outputs for all 3 IR system.
4. q1.py - This is file for first question in the assignment which majorily doing text preprocessing and cleaning.
5. q2.py - This contains necessary function required for ques. 2 having all the IR systems and their helper functions.
6. main.py - This is the main file which combines whole assignments.
7. run.sh - This is the file that contains all the variable parameters mentioned in the below section.
8. Makefile - There are two commands in the makefile one is "install", "run"
    a.  make install - install all the required packages and download the drive files (please follow the drive link if you're not able to download from the make install)
    b.  make run - will run the whole assignments 

*** run.sh is the top-level script that runs the entire assignment. ***

### To run the entire assignment, go to home directory where this README file is there and use following command
##  $ make install 
##  $ make run 

These are the variables that I'm passing as an arguments in the program. [ change accordingly ]

1. top_k = 5 - top k documents will be retrieved
2. corpus = "./data/english-corpora/" - Corpus path
3. vocab_path = './utils/vocab_doc_wise_stemming.npy' - This is the vocab file path (in numpy format) document wise which contains a dictionary where keys are docID and values are the vocab present in that documents which I have already generated required in all 3 IR systems. This takes time to build so if you still want, you can build your own. There is one flag named vocab_flag. Put this variable equal to 1. Otherwise you can use my generated vocab_doc_wise_stemming.npy present in the utils folder.
4. postings_path="./utils/postings_list.pkl" - This is the postings list already generated by myself required in all 3 IR systems. This also takes time to build if you want to build put postings_flag=1.
5. doc_vector_path='./utils/sparse_matrix_doc_vectors.npz' - This is Doc vectors file required for TF-IDF system. This also takes time to build so if you want to build put vector_flag=1.
6. query_file='./data/query.txt' - This is the query path you need to give if you want to try my assignment on someone else query.
7. out_folder='./output/' - This is the output folder path.
8. vocab_flag=0 - This is for creating own vocab_doc_wise_stemming
9. postings_flag=0 - This is for creating own postings lists
10. vector_flag=0 - This is for creating own docments vectors.


#### Preprocessing #####################################

1. Text is splitted by '\t' first.
2. Remove extra spaces
3. Tokenize the strings
4. Remove Punctuations from tokenize words
5. Remove Number
6. Remove Double quotations from tokens
7. Replace URL with url tag
8. Remove ascents from string using decode like A&deg;
9. Split camelCase word into 'camel' and 'Case'
10. Remove number from token


#### Boolean IR System #################################

This system takes an average 0.0032 sec to process one query.

1. Tokenize the query
2. Convert infix query expression to postfix query expression using stack approach
        a. Check if the given expression is balanced or not
        b. Check is there any extra parenthesis in the expression
3. Processing two operator only in the query **\&**(and) , **\|** (or) and **\~**(negation) and giving higeher precedence to the former
4. Using **snowball_stemmer** as a stemmer algorithm to find the stem word in the given query
5. Generate binary vector based on document size and consider negation sign as well while processing
6. Find document which contains the query word using **find_matched_doc** function and return a binary vector that shows which document contains that word
7. Remove stop words from query

#### TF-IDF IR System ###################################

This system takes an average 22 sec to process one query.

1. Tokenize the query first and remove the stopwords from the query and also remove nonASCII character
2. Find the query vector where each dimension represents freq. of token present in the query
3. For fast query processing, I have created one doc_vectors which is of shape (doc_size, total vocab) previously and saved their vectors in sparse_matrix_doc_vectors.npz
4. In doc vectors, rows represent all the documents and columns represent all the vocab present in all the documents 
5. Doc vectors are sparse matrix inorder to save space I used scipy.sparse.matrix to save all doc vectors.
6. Return top-k documents only.
7. I have also tried to prepare champion lists that is nothing but for each vocab in the documents find the rank of documents corresponding to them. But
    it was taking too much space more than 24 GB) and also taking more than 24 hrs to prepare those lists. But I have provided the code for the same.
8. While calculating similarity score we don't need to normalize the query vector as even without normalization, product of V_q and v_d are much higher.

#### BM-25 IR System ####################################

This system takes an average 0.018 sec to process one query.

1. Tokenize the query into tokens and remove the stop words and also remove if there's any non-ascii characters
2. Get local weight by modified term frequency formula $$\frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$
3. Get global weight by inverse doc frequency as the priors aren't given by given formula $$\log \frac{n}{df_t}$$
4. Get RSVd score using below formula and based on this score, select top k documents $$RSVd = \sum_{\forall t \in q} \left(\log \frac{n}{df_t}\right) . \frac{(k_1+1)tf_d}{k_1(1-b+b\frac{L_d}{L_avg}) + tf_d}$$


# Incase you face any issue in running the code, just let me know here - rahulkumar21@iitk.ac.in
