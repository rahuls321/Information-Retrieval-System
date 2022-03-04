install:
	pip install -r requirements.txt
	wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1suxXJpFDS3kbc3ndTppo9G0GAAdeTVsA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1suxXJpFDS3kbc3ndTppo9G0GAAdeTVsA" -O ./utils/postings_list.pkl && rm -rf /tmp/cookies.txt
	wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kHeVgl36O1_oO3vFIWcQLBv-z5V1mHaR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kHeVgl36O1_oO3vFIWcQLBv-z5V1mHaR" -O ./utils/sparse_matrix_doc_vectors.npz && rm -rf /tmp/cookies.txt
	wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1l3OQA6RSnWk2EWIpAuRTIDNcDzgue2nQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1l3OQA6RSnWk2EWIpAuRTIDNcDzgue2nQ" -O ./utils/vocab_doc_wise_stemming.npy && rm -rf /tmp/cookies.txt

run:
	./run.sh
