import os

directory = "./dataset/data/xsum-extracts-from-downloads"

inputFile = "./dataset/data/url_data/stanford-inputlist.txt"

with open(inputFile, 'w') as f:
    for filename in os.listdir(directory):
        f.write(filename)
        f.write('\n')
