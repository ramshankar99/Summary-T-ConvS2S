'''
Training the lda model 
'''
import gensim
import json
import nltk
nltk.download('stopwords')
import string
from nltk.decorators import memoize
import numpy as np
import sys
import os

if len(sys.argv) == 3:
    TOTAL_NUM_TOPICS = int(sys.argv[1])
    TOTAL_NUM_ITER = int(sys.argv[2])
else:
    TOTAL_NUM_TOPICS = 512
    TOTAL_NUM_ITER = 1000

split_dict = "./dataset/data/XSum-DATA-SPLIT.json"
smartstopwords = "./dataset/data/smart-stopwords.txt"
output_dir = './dataset/data/xsum-lda-train-document-lemma-topic-' + str(TOTAL_NUM_TOPICS) + '-iter-' + str(TOTAL_NUM_ITER)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Load JSON File : training, dev and test splits.
with open(split_dict) as json_data:
    train_dev_test_dict = json.load(json_data)
    
# Corpus
corpusdir = "./dataset/data/xsum-data-preprocessed"

## Stopwords
file_stop = open('C:\\Users\\rams9\\AppData\\Roaming\\nltk_data\\corpora\\stopwords\\english')
data = file_stop.read()
stopwords = set(data)
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
    corpus = nltk.corpus.PlaintextCorpusReader(corpusdir + "/document-lemma", r'.*document-lemma')
    for fileidonly in train_dev_test_dict["train"]:
        fileid = fileidonly + ".document-lemma"
        yield list(normalize_words("".join(corpus.raw(fileid)).split()))
        
if __name__ == "__main__":
    
    # Build id2word on the training data
    print("Prepare id2word on the training data")
    id2word = gensim.corpora.Dictionary(documents())
    # print id2word
    corpus = [id2word.doc2bow(doc) for doc in documents()]
    # print corpus[0]
    print() 
    
    # Prepare Gensim data, train LDA model
    print("Prepare Gensim data for the training data")
    gensim.corpora.MmCorpus.serialize(output_dir + '/corpus.mm', corpus)
    mm = gensim.corpora.MmCorpus(output_dir + '/corpus.mm')
    print(mm)
    print()
    print("Start training the LDA model")
    # lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=10, update_every=1, passes=20)
    lda = gensim.models.ldamulticore.LdaMulticore(corpus=mm, id2word=id2word, num_topics=TOTAL_NUM_TOPICS, iterations=TOTAL_NUM_ITER, workers=20, minimum_probability=None, minimum_phi_value=None)
    # Save
    print()
    print("Saving the LDA model")
    lda.save(output_dir + '/lda.model')
    print()
    
    # Print topics
    print( "Start printing the learned topics")
    for topic in lda.print_topics(TOTAL_NUM_TOPICS, num_words=10):
        print(topic)
        print()

    # # Example output: topic distribution of the first document
    # print "Topic distribution for the first document: "
    # print lda[corpus[0]]
    # print
