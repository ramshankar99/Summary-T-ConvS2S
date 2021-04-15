# Dataset Creation

Order to run files:

1. pip3 install -r requirements.txt
2. download-bbc-articles.py
3. parse-bbc-articles.py
4. helper-list-extracted-data-files.py
5. process-corenlp-annotate.py
6. helper-xsum-data-train-val-test-split.py
7. process-corenlp-xml-data.py

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

# Training 

## Data Preprocessing 
Generate source and target dictionary files. In this case, both files are identical (due to "--joined-dictionary"). It operates on the raw format data.
```
TEXT= {path to xsum_data_topic_convs2s dir}
!python ./dataset/scripts/XSum-Topic-ConvS2S/preprocess.py --source-lang document \
                                         --target-lang summary \
                                         --trainpref $TEXT/train \
                                         --validpref $TEXT/validation \
                                         --testpref $TEXT/test \
                                         --destdir $TEXT \
                                         --joined-dictionary \
                                         --nwordstgt 50000 \
                                         --nwordssrc 50000 \
                                         --output-format raw
```

## Model Training 

The model requires GPU for training. Check usage with -h for changing variant and hyperparameters
```
save_directory = "./dataset/checkpoints-topic-convs2s"
CUDA_VISIBLE_DEVICES=1 
!python ./dataset/scripts/XSum-Topic-ConvS2S/train.py $TEXT --source-lang document \
                                                            --target-lang summary \
                                                            --doctopics doc-topics \
                                                            --max-sentences 32 \
                                                            --arch fconv \
                                                            --variant 2 \
                                                            --criterion label_smoothed_cross_entropy \
                                                            --max-epoch 200 \
                                                            --clip-norm 0.1 \
                                                            --lr 0.10 \
                                                            --dropout 0.2 \
                                                            --save-dir {save_directory} \
                                                            --no-progress-bar \
                                                            --log-interval 10
```


# ROUGE

```
!python path/eval_rouge.py --summary {system_summary_file} --mod_sum {model_summary_file}
```
Takes a single txt file with generated summary and a file with the model gold summary file and evaluates P, R, F on rouge-1, rouge-2, rouge-l
Sample Output
```
rouge-1:	P: 30.00	R: 37.50	F1: 33.33
rouge-2:	P: 11.11	R: 14.29	F1: 12.50
rouge-l:	P: 26.15	R: 31.50	F1: 28.58
```
