'''
script to get annotated xml files from data files
using nlp core 
'''
import argparse
import stanza
import os
from stanza.server import CoreNLPClient # Client Module
import time

# Construct a CoreNLPClient with some basic annotators, a memory allocation of 4GB, and port number 9001
def createClient():
    '''
    Starts NLP core client
    '''
    client = CoreNLPClient(
        annotators=['tokenize','ssplit', 'pos', 'lemma'], 
        memory='4G', 
        endpoint='http://localhost:9001',
        be_quiet=True)
    print(client)

    client.start()
    time.sleep(10)
    return client

def closeClient(client):
    '''
    Closes client
    '''
    client.stop( )
    time.sleep(10)

def processfiles(client, fileList, dataset_dir, output_dir):
    
    with open(fileList, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            fp = open(dataset_dir + line, "r")
            text = fp.read()
            doc = client.annotate(text, properties={"outputFormat" : "xml"})
            id_ = line.split("/")[-1]
            fo = open(output_dir + id_ + ".xml", "w")
            fo.write(doc)
            fp.close()
            fo.close()

if __name__ == "__main__":
    corenlp_dir = './corenlp'
    dataset_dir = './dataset/data/xsum-extracts-from-downloads/'
    # Install corenlp if not present in the current folder
    if not os.path.exists(corenlp_dir):
        stanza.install_corenlp(dir=corenlp_dir)

    # Set the CORENLP_HOME environment variable to point to the installation location
    os.environ["CORENLP_HOME"] = corenlp_dir
    
    parser = argparse.ArgumentParser(description='Create xml files')
    parser.add_argument('--fileList', type=str, default="./dataset/data/url_data/stanford-inputlist.txt", help='list of .data files in the format xsum-extracts-from-downloads/{id}.data')
    parser.add_argument('--output', type=str, default="./dataset/data/xsum-corenlp-xml-output/", help = "Directory where xml files will be stored")
    args = parser.parse_args()

    fileList = args.fileList
    output_dir = args.output

    try:
        print('Creating the output directory.')
        os.mkdir(output_dir)
    except WindowsError:
        print('Output directory already exists.')

    # Create client 
    client = createClient()

    # Process files
    processfiles(client, fileList, dataset_dir, output_dir)

    # Close client
    closeClient(client)

  





