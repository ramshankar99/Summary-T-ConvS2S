import gensim
import json
import nltk
import string
from nltk.decorators import memoize
import numpy as np
import os
import sys
import os.path

if len(sys.argv) == 3:
  TOTAL_NUM_TOPICS = int(sys.argv[1])
  TOTAL_NUM_ITER = int(sys.argv[2])
else:
  TOTAL_NUM_TOPICS = 512
  TOTAL_NUM_ITER = 1000


smartstopwords = "./dataset/scripts/smart-stopwords.txt"
ldadir = './dataset/data/lda-train-document-lemma-topic-'+str(TOTAL_NUM_TOPICS)+'-iter-'+str(TOTAL_NUM_ITER)
    
# Corpus
corpusdir = "./dataset/data/xsum-data-preprocessed"

## Stopwords
file_stop=open('/root/nltk_data/corpora/stopwords/english')
data=file_stop.read()
stopwords   = set(data)
# Smart Stopwords
stopwords = stopwords.union(set([item.strip() for item in open(smartstopwords).readlines() if len(item.strip()) != 0]))

## Punctuations
punctuation = string.punctuation
extra_punctuation = ["\'\'", "``", "-rrb-", "-lrb-", "\'s"]

@memoize
def normalize(token):
    token = token.lower()
    # token = lemmatizer.lemmatize(token)
    return token

def normalize_words(words):
    for token in words:
        token = normalize(token)
        if token not in stopwords and token not in punctuation and token not in extra_punctuation:
            yield token

def documents():
    corpus = nltk.corpus.PlaintextCorpusReader(corpusdir+"/document-lemma", r'.*document-lemma')
    for filename in corpus.fileids():
        fileid = filename.split(".")[0]
        yield [fileid, list(normalize_words("".join(corpus.raw(filename)).split()))]

if __name__ == '__main__':

    # Load LDA model
    print ("Loading LDA model from "+ldadir+ "...")
    lda = gensim.models.ldamulticore.LdaMulticore.load(ldadir+'/lda.model', mmap='r')

    outputdir = corpusdir+"/document-lemma-topic-"+str(TOTAL_NUM_TOPICS)+"-iter-"+str(TOTAL_NUM_ITER)
    os.system("mkdir -p "+outputdir)
    
    print ("Start decoding in "+outputdir+"...")
    count = 0
    for fileid, doc in documents():

        if os.path.isfile(outputdir+"/"+fileid+".doc-topics"):
            continue

        bow = lda.id2word.doc2bow(doc)
        # print bow
        topics = lda.get_document_topics(bow, per_word_topics=True, minimum_probability=0, minimum_phi_value=0)
        
        # topic_dict = {}
        outstr = ""
        for item in topics[0]:
            # topic_dict[item[0]] = 1
            outstr += (str(item[0]) + "\t" + str(item[1])+"\n")
            
        foutput = open(outputdir+"/"+fileid+".doc-topics", "w")
        foutput.write(outstr)
        foutput.close()

        count += 1
        if count%10000 == 0:
            print( count)