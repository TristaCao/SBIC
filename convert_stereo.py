import csv
import json 
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

random.seed(42)

input_file = '../mlm/StereoSet/data/dev.json'
out = []
eid = 0
with open(input_file, "r") as f:
    data = json.load(f)
    examples = data['data']['intrasentence']
    for example in examples:
        sents = example['sentences']
        context = example['context']
        for sent in sents:
            label = 1 if sent['gold_label']=='stereotype' else 0
            s = sent['sentence'].lower()
            out.append([eid, s, label])
            eid += 1
    examples = data['data']['intersentence']
    for example in examples:
        sents = example['sentences']
        context = example['context']
        for sent in sents:
            label = 1 if sent['gold_label']=='stereotype' else 0
            s = context+' '+sent['sentence'].lower()
            out.append([eid, s, label])
            eid += 1

out = np.array(out)
train, test = train_test_split(out, test_size=0.2, random_state=42)
dev, test = train_test_split(test, test_size=0.5, random_state=42)

in_file = 'SBIC/SBIC.v2.agg.trn.csv'
strain = []
with open(in_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    text_colname = "post"
    for data in reader:
        x_vals = data[text_colname].replace('\n', '').lower()
        ids = int(data["id"])+ eid
        y_vals = float(data["hasBiasedImplication"])
        y_vals = 1 if y_vals == 0 else 0
        strain.append([ids, x_vals, y_vals])
in_file = 'SBIC/SBIC.v2.agg.dev.csv'
sdev = []
with open(in_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    text_colname = "post"
    for data in reader:
        x_vals = data[text_colname].replace('\n', '').lower()
        ids = int(data["id"])+eid
        y_vals = float(data["hasBiasedImplication"])
        y_vals = 1 if y_vals == 0 else 0
        sdev.append([ids, x_vals, y_vals])
in_file = 'SBIC/SBIC.v2.agg.tst.csv'
stest = []
with open(in_file, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    text_colname = "post"
    for data in reader:
        x_vals = data[text_colname].replace('\n', '').lower()
        ids = int(data["id"])+eid
        y_vals = float(data["hasBiasedImplication"])
        y_vals = 1 if y_vals == 0 else 0
        stest.append([ids, x_vals, y_vals])

train =np.concatenate((train, np.array(strain)))
dev =np.concatenate((dev, np.array(sdev)))
test =np.concatenate((test, np.array(stest)))

np.random.shuffle(train)
np.random.shuffle(dev)
np.random.shuffle(test)

with open('all_train.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["id", 'sentence', 'label'])
    write.writerows(train)
with open('all_dev.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["id", 'sentence', 'label'])
    write.writerows(dev)
with open('all_test.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(["id", 'sentence', 'label'])
    write.writerows(test)



