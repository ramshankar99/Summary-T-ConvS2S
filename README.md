# Summary-T-ConvS2S

An implementation of [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization](https://arxiv.org/abs/1808.08745)

# XSum Dataset

Follow the steps mentioned in this [README](https://github.com/ramshankar99/Summary-T-ConvS2S/blob/master/dataset/README.md) for
1. Generating the XSum dataset starting from bbc Urls
2. Training the LDA Model from scratch 
3. Decoding word-topics and doc-topics using the LDA model
4. Data Processing 

# Training 

## Data Preprocessing 
Generate source and target dictionary files. In this case, both files are identical (due to "--joined-dictionary"). It operates on the raw format data.
```
TEXT= {path to xsum_data_topic_convs2s dir}
!python ./XSum-Topic-ConvS2S/preprocess.py --source-lang document \
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
save_directory = "./checkpoints-topic-convs2s"
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
Take txt files with generated summaries and a file with the corresponding model gold summaries and evaluates P, R, F on rouge-1, rouge-2, rouge-l
Sample Output
```
rouge-1:	P: 30.00	R: 37.50	F1: 33.33
rouge-2:	P: 11.11	R: 14.29	F1: 12.50
rouge-l:	P: 26.15	R: 31.50	F1: 28.58
```
