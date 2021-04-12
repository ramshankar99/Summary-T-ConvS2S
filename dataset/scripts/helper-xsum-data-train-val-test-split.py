import json

inputFile = "./dataset/data/url_data/stanford-inputlist.txt"

filenames = []
with open(inputFile, 'r') as f:
    lines = f.readlines()
    for line in lines:
        filename = line.split('.')[0]
        filenames.append(filename)

n = len(filenames)
train_len = int(0.9 * n)
val_len = int((n - train_len) / 2)
test_len = n - train_len - val_len

# XSum-TRAINING-DEV-TEST-SPLIT-90-5-5
train = filenames[:train_len]
validation = filenames[train_len:train_len + val_len]
test = filenames[train_len + val_len:train_len + val_len + test_len]

jsondata = {"train": train, "validation": validation, "test": test}

with open('./dataset/data/XSum-DATA-SPLIT.json', 'w') as fp:
    json.dump(jsondata, fp)
