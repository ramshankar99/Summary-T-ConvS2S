import gensim
import json
import nltk
import string
from nltk.decorators import memoize
import numpy as np
import os
import sys

if len(sys.argv) == 3:
  TOTAL_NUM_TOPICS = int(sys.argv[1])
  TOTAL_NUM_ITER = int(sys.argv[2])
else:
  TOTAL_NUM_TOPICS = 512
  TOTAL_NUM_ITER = 1000

ldadir = './dataset/data/lda-train-document-lemma-topic-'+str(TOTAL_NUM_TOPICS)+'-iter-'+str(TOTAL_NUM_ITER)
    
if __name__ == '__main__':

    # Load LDA model
    print ("Loading LDA model from "+ldadir+ "...")
    lda = gensim.models.ldamulticore.LdaMulticore.load(ldadir+'/lda.model', mmap='r')
    print (len(lda.id2word))

    count = 0
    for id in range(len(lda.id2word)):
        print (id, lda.id2word[id].encode("UTF-8"), " ".join([str(item[0])+":"+str(item[1]) for item in lda.get_term_topics(id, minimum_probability=0)]))
        count += 1
        