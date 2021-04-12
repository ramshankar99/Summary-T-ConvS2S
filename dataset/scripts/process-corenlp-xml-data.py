
# Run with python 3

import argparse
import json
import os
# import xml.etree.ElementTree as ET
import lxml.etree as ET
import shutil

if __name__ == "__main__":
    
	parser = argparse.ArgumentParser(description='Parse xml files')
	parser.add_argument('--stanford_dir', type=str, default="./dataset/data/xsum-corenlp-xml-output")
	parser.add_argument('--split_dict', type=str, default="./dataset/data/XSum-DATA-SPLIT.json")
	parser.add_argument('--output_dir', type=str, default="./dataset/data/xsum-data-preprocessed")
	args = parser.parse_args()

	stanford_directory = args.stanford_dir
	split_file = args.split_dict
	output_directory = args.output_dir

	# Load split file
	split_dict = json.loads(open(split_file).read())
	data_types = ["test", "validation", "train"]

	try:
		if os.path.exists(output_directory): shutil.rmtree(output_directory)
		print('Creating the output directory.')
		# Creating Directories
		os.mkdir(output_directory)
	
		os.mkdir(output_directory + "/document")
		os.mkdir(output_directory + "/document-lemma")
		os.mkdir(output_directory + "/summary")
	except WindowsError:
		print('Output directory already exists.')
		exit(0)
  
	fm = open('./dataset/data/url_data/final-data-failed-decoding.txt', 'w')

	count = 1
	for dtype in data_types:
		print(dtype)
		fm.write(dtype + '\n')
		
		for orgfileid in split_dict[dtype]:

			if (os.path.isfile(output_directory + "/document/" + orgfileid + ".document") and
				os.path.isfile(output_directory + "/document-lemma/" + orgfileid + ".document-lemma") and
				os.path.isfile(output_directory + "/summary/" + orgfileid + ".summary")):
				continue

			print(orgfileid)
			
			stanfordfile = stanford_directory + "/" + orgfileid + ".data.xml"
			
			doc_sentences = []
			doc_sentlemmas = []
			# process xml file
			try:
				tree = ET.parse(stanfordfile, ET.XMLParser(encoding='utf-8', recover=True))
			except ET.XMLSyntaxError:
				fm.write(orgfileid)
			root = tree.getroot()
			
			for sentences in root.iter('sentences'):
				for sentence in sentences.iter('sentence'):
					sentence_tokenized = []
					sentence_lemmatized = []
					for token in sentence.iter('token'):
						word = token.find('word').text
						sentence_tokenized.append(word)
						lemma = token.find('lemma').text
						sentence_lemmatized.append(lemma)
      
					sent_tok = " ".join(sentence_tokenized)
					sent_lem = " ".join(sentence_lemmatized)
     
					substring = "[ XSUM ] INTRODUCTION [ XSUM ]"
					lem_substring = "[ XSUM ] introduction [ XSUM ]"
					rb_substring = "[ XSUM ] RESTBODY [ XSUM ]"
					rb_lem_substring = "[ XSUM ] restbody [ XSUM ]"
      
					if sent_tok.find(substring) != -1:
						pos = sent_tok.find(substring)
						pos_lem = sent_lem.find(lem_substring)
						if pos != 0:
							if sent_tok.find(rb_substring) == -1:
								doc_sentences.append(sent_tok[:pos-1])
								doc_sentences.append(sent_tok[pos:])
								doc_sentlemmas.append(sent_lem[:pos_lem-1])
								doc_sentlemmas.append(sent_lem[pos_lem:])
							else:
								pos_rb = sent_tok.find(rb_substring)
								pos_rb_lem = sent_lem.find(rb_lem_substring)
								doc_sentences.append(sent_tok[:pos-1])
								doc_sentences.append(sent_tok[pos:pos_rb-1])
								doc_sentences.append(sent_tok[pos_rb:])
								doc_sentlemmas.append(sent_lem[:pos_lem-1])
								doc_sentlemmas.append(sent_lem[pos_lem:pos_rb_lem-1])
								doc_sentlemmas.append(sent_lem[pos_rb_lem:])
								
						else:
							pos_2 = sent_tok.find(rb_substring)
							pos_2_lem = sent_lem.find(lem_substring)
							doc_sentences.append(sent_tok[:pos_2-1])
							doc_sentences.append(sent_tok[pos_2:])
							doc_sentlemmas.append(sent_lem[:pos_2_lem-1])
							doc_sentlemmas.append(sent_lem[pos_2_lem:])
					else:
						doc_sentences.append(sent_tok)
						doc_sentlemmas.append(sent_lem)
					
			# print("S1:", doc_sentences[0])
			# print("S2:", doc_sentences[1])
			# print("S3:", doc_sentences[2])
			# print(len(doc_sentences))
			# input()

			# Extract data
			modeFlag = None

			restbodydata = []
			restbodylemmadata = []
			summarydata = []
			
			allcovered = 0
			for doc_sent, doc_sentlemma in zip(doc_sentences, doc_sentlemmas):
				i =0
				if "[ XSUM ] URL [ XSUM ]" in doc_sent:
					modeFlag = "URL"
					allcovered += 1
				elif "[ XSUM ] INTRODUCTION [ XSUM ]" in doc_sent:
					modeFlag = "INTRODUCTION"
					allcovered += 1
				elif "[ XSUM ] RESTBODY [ XSUM ]" in doc_sent:
					modeFlag = "RESTBODY"
					allcovered += 1

				if modeFlag == "RESTBODY":
				    if i == 0:
				        restbodydata.append(doc_sent)
				        restbodylemmadata.append(doc_sentlemma)
				        i+= 1
				    else:
				        restbodydata.append(doc_sent)
				        restbodylemmadata.append(doc_sent)
				elif modeFlag == "INTRODUCTION":
					summarydata.append(doc_sent[30:]) # Starting after [ XSUM ]

			if allcovered != 3:
				print("Some information missing", stanfordfile)
				fm.write(orgfileid + '\n')
				continue
				# print("\n".join(doc_sentences))
			print("All 3 generated!")

			restbodydata[0] = restbodydata[0][26:]
			foutput = open(output_directory+"/document/"+orgfileid+".document", "w")
			foutput.write("\n".join(restbodydata)+"\n")
			foutput.close()
			
			restbodylemmadata[0] = restbodylemmadata[0][26:]
			foutput = open(output_directory+"/document-lemma/"+orgfileid+".document-lemma", "w")
			foutput.write("\n".join(restbodylemmadata)+"\n")
			foutput.close()

			# summary data
			foutput = open(output_directory+"/summary/"+orgfileid+".summary", "w")
			foutput.write("\n".join(summarydata)+"\n")
			foutput.close()
			
			if count%1000 == 0:
				print(count)
			count += 1
			
			# fm.close()
			# exit(0)
	fm.close()
			