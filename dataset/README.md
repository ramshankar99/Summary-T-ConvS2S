# Dataset Creation

> change directory to path/Summary-T-ConvS2S/ for convenience 

Order to run files:

1. pip3 install -r requirements.txt
2. download-bbc-articles.py
3. parse-bbc-articles.py
4. helper-list-extracted-data-files.py
5. process-corenlp-annotate.py
6. helper-xsum-data-train-val-test-split.py
7. process-corenlp-xml-data.py

> Note: You can specify the number of Urls to download for download-bbc-articles.py by changing the variable URLS_TO_DOWNLOAD in the .py file 
> Currently set to 1000

# LDA Training and Decoding

Train LDA model with 
```
!python path/lda-gensim-training-document-lemma.py > ./dataset/data/xsum-lda-train/lda-train.log
```
Decode the word topics and document topics using trained LDA model
1. Word Topics
```
!python path/lda-gensim-decoding-wordtopicdist.py 
```
Log saved in ./dataset/data/xsum-lda-train/document-lemma-topic-512-iter-1000/word_term_topics.log
2. Document Topics 
```
!python path/lda-gensim-decoding-document-lemma.py
```
Saves doc-topic files in ./dataset/data/xsum-data-preprocessed/document-lemma-topic-512-iter-1000/{id}.doc-topics
<br><br>
Check visualise_lda.html for an interactive summary of the lda training outcome 

# Preprocess 

```
python path/processing.py
```
It generates the following files in the "data-topic-convs2s" directory:

```
train.document, train.summary, train.document-lemma and train.doc-topics
validation.document, validation.summary, validation.document-lemma and validation.doc-topics
test.document, test.summary, test.document-lemma and test.doc-topics
dict.document-lemma.lda.txt
```
Lines in document, summary, document-lemma and doc-topics files are paired as (input document, output summary, input lemmatized document, document topic vector).

<hr>
