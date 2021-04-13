'''
Generating word to topic 
Usage: !python /path/lda-gensim-decoding-wordtopicdist.py 
Saves log in ./dataset/data/xsum-lda-train/document-lemma-topic-512-iter-1000/word_term_topics.log
'''
import gensim
# import json
# import nltk
# import string
# from nltk.decorators import memoize
# import numpy as np
# import os
import sys

if len(sys.argv) == 3:
    TOTAL_NUM_TOPICS = int(sys.argv[1])
    TOTAL_NUM_ITER = int(sys.argv[2])
else:
    TOTAL_NUM_TOPICS = 512
    TOTAL_NUM_ITER = 1000

lda_dir = './dataset/data/xsum-lda-train/document-lemma-topic-' + str(TOTAL_NUM_TOPICS) + '-iter-' + str(TOTAL_NUM_ITER)
    
if __name__ == '__main__':

    with open(lda_dir + '/word_term_topics.log', 'w+') as f:

        # Load LDA model
        f.write("Loading LDA model from " + lda_dir + "...\n")
        lda = gensim.models.ldamulticore.LdaMulticore.load(lda_dir + '/lda.model', mmap='r')
        f.write(str(len(lda.id2word)) + '\n')

        count = 0
        for id in range(len(lda.id2word)):
            f.write(str(id) + ' ')
            f.write(str(lda.id2word[id].encode("UTF-8")) + ' ')
            f.write(" ".join([str(item[0])+":"+str(item[1]) for item in lda.get_term_topics(id, minimum_probability=0)]) + '\n')
            count += 1
        