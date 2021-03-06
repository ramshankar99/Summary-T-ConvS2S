import sys
import os
import json

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

bbc_tokenized_stories_dir = "./dataset/data/xsum-data-preprocessed"
lda_word_topics = "./dataset/data/xsum-lda-train/document-lemma-topic-512-iter-1000/word_term_topics.log"
finished_files_dir = "./dataset/data/data-topic-convs2s"
split_dict = "./dataset/data/XSum-DATA-SPLIT.json"

# Load JSON File : training, dev and test splits.
with open(split_dict) as json_data:
  train_dev_test_dict = json.load(json_data)
  
def read_text_file(text_file):
  '''
  read lines from text file 
  return: list of lines with trailing and leading spaces removed
  '''
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines

def fix_missing_period(line):
  '''Adds a period to a line that is missing a period'''
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def get_data_from_file(story_file):
  '''
  Return: single string article with removed trailing and leading spaces and fixed periods 
  '''
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Make article into a single string
  article = ' '.join(lines)

  return article

def get_doctopic_data_from_file(doctopicfile, topic_list):
  '''
  doctopicfile: document and topic file from lda
  topic_list: list of topics 
  '''
  lines = read_text_file(doctopicfile)
  if len(lines) != 512: # we use 512 topics 
    print(doctopicfile)
    print("Not all topics given probability.")
    exit(0)
  data = []
  for line in lines:
    ldata = line.split()
    data.append(ldata[0]+":"+ldata[1])
  return ",".join(data)

def write_to_bin(data_type, out_file_rb, out_file_fs, out_file_doctopics, out_file_document_lemma, word_topic_dict_final, topic_list, count=0):
  
  """Reads all the bbids and write them to out file."""
  print ("Making text file for bibids listed as {}...".format(data_type))
  
  bbcids = train_dev_test_dict[data_type]
  num_stories = len(bbcids)
  print(num_stories)

  rb_foutput = open(out_file_rb, "w")
  fs_foutput = open(out_file_fs, "w")
  doctopics_foutput = open(out_file_doctopics, "w")
  document_lemma_foutput = open(out_file_document_lemma, "w")
  
  for idx,s in enumerate(bbcids):
    
    if idx % 1000 == 0:
      print("Writing story {} of {}; {} percent done".format(idx, num_stories, float(idx)*100.0/float(num_stories)))

    # Input Files
    documentfile = bbc_tokenized_stories_dir + "/document/" + s + ".document"
    summaryfile = bbc_tokenized_stories_dir + "/summary/" + s + ".summary"
    documentfile_lemma = bbc_tokenized_stories_dir + "/document-lemma/" + s + ".document-lemma"
    doctopicfile = bbc_tokenized_stories_dir + "/document-lemma-topic-512-iter-1000/" + s + ".doc-topics"
            
    # Get the strings to write to .bin file
    abstract = get_data_from_file(summaryfile)
    article = get_data_from_file(documentfile)
    article_lemma = get_data_from_file(documentfile_lemma)
 
    article_word_list = article.strip().split()
    article_lemma_list = article_lemma.strip().split()

    # print("A ",len(article_word_list), article_word_list)
    # print("B ", len(article_lemma_list), article_lemma_list)

    if len(article_word_list) != len(article_lemma_list ):
      print(idx, s)
      print("Word count and lemma count did not match.")
      count +=1
      # exit(0)

    article = article_word_list[:400]
    article_lemma = article_lemma_list[:400]
    
    # Get Document Topic Vector, in this case all 1 
    doctopic = get_doctopic_data_from_file(doctopicfile, topic_list)
    # doctopic = ",".join([(str(i)+":1.0") for i in range(512)])

    article = " ".join(article)
    article_lemma = " ".join([lemma if lemma in word_topic_dict_final else "UNK" for lemma in article_lemma])
    
    # print(idx, s)
    # print(abstract)
    # print(article)
    # print(" ".join(article_lemma))
    # print(doctopic)
    # print(lemmatopic)
    
    rb_foutput.write(article+"\n")
    fs_foutput.write(abstract+"\n")
    doctopics_foutput.write(doctopic+"\n")
    document_lemma_foutput.write(article_lemma+"\n")

    # exit(0)
    
  rb_foutput.close()
  fs_foutput.close()
  doctopics_foutput.close()
  document_lemma_foutput.close()

  print("WORD LEMMA PROBLEM: ",count)
  
  print("Finished writing file:\n{}\n{}\n{}\n{}\n".format(out_file_rb, out_file_fs, out_file_doctopics, out_file_document_lemma))

if __name__ == '__main__':

  # Create some new directories
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read word_topic vector
  print("Reading word_topic vector")
  word_topic_dict = {} # creates dict for each word topic:topicprob
  wordid_word_dict = {} # word id to word mapping where word in format "b'academy'" 
  
  min_word_topic_prob = 100
  count = 0 # To skip the first two lines
  with open(lda_word_topics) as f:
    for line in f:
      count += 1
      if count > 2:
        ldata = line.split() # len is 17
        wordid = int(ldata[0]) # word id
        word = ldata[1] # word in utf-8 "b'academy'" 
        word_topic_dict[word] = {}
        wordid_word_dict[wordid] = word
        for topic_prob in ldata[2:]:
          topic_prob_data = topic_prob.split(":")
          word_topic_dict[word][topic_prob_data[0]] = topic_prob_data[1]
          if float(topic_prob_data[1]) < min_word_topic_prob:
            min_word_topic_prob = float(topic_prob_data[1])
        

  print("min_word_topic_prob: "+str(min_word_topic_prob))
  str_min_word_topic_prob = str(min_word_topic_prob)

  topic_list = [str(i) for i in range(512)] # list 0 to 511
  empty_word_topic = [str_min_word_topic_prob for i in topic_list] # assign min prob to all topics 
  
  flemma_dict = open(finished_files_dir+"/dict.document-lemma.lda.txt", "w")
  flemma_dict.write("UNK "+" ".join(empty_word_topic)+"\n") # assign min prob to all topics for UNK token
  wordids = wordid_word_dict.keys() # word id numbers 0,1,2....num_words-1
  wordids = list(wordids)
  wordids.sort()
  for wordid in wordids:
    word = wordid_word_dict[wordid]
    # Assign topic vector using dictionary and default values
    topic_vector = [word_topic_dict[word][i] if i in word_topic_dict[word] else str_min_word_topic_prob for i in topic_list]
    flemma_dict.write(word+" "+" ".join(topic_vector)+"\n")
  flemma_dict.close()
              
  # Read the tokenized stories, do a little postprocessing then write to text files

  write_to_bin("test", os.path.join(finished_files_dir, "test.document"), os.path.join(finished_files_dir, "test.summary"),
               os.path.join(finished_files_dir, "test.doc-topics"), os.path.join(finished_files_dir, "test.document-lemma"),
               word_topic_dict, topic_list)
  
  write_to_bin("validation", os.path.join(finished_files_dir, "valid.document"), os.path.join(finished_files_dir, "valid.summary"),
               os.path.join(finished_files_dir, "valid.doc-topics"), os.path.join(finished_files_dir, "valid.document-lemma"),
               word_topic_dict, topic_list)
  
  write_to_bin("train", os.path.join(finished_files_dir, "train.document"), os.path.join(finished_files_dir, "train.summary"),
               os.path.join(finished_files_dir, "train.doc-topics"), os.path.join(finished_files_dir, "train.document-lemma"),
               word_topic_dict, topic_list)