import os
from collections import defaultdict
import json
import sys
from sklearn.metrics import roc_auc_score
import jsonlines
annotations_path = "C:\\Users\\taoli1\code\\MultiModal\\ofa_proj\\marvl-code-forked\\data\\zh\\annotations_machine-translate\\marvl-zh_gmt.jsonl"

def load_annotations():
    items = []
    with jsonlines.open(annotations_path) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_id_0"] = annotation["left_img"].split("/")[-1].split(".")[0]
            dictionary["image_id_1"] = annotation["right_img"].split("/")[-1].split(".")[0]
            dictionary["question_id"] = count
            dictionary["idx"] =  dictionary["image_id_0"] + '##' +  dictionary["image_id_1"]

            dictionary["sentence"] = str(annotation["caption"])
            dictionary["labels"] = [int(annotation["label"])]
            dictionary["scores"] = [1.0]
            items.append(dictionary)
            # print(dictionary["sentence"])
            count += 1
            if count < 2:
                print("loading_annotations: ")
                print(dictionary)
    return items
annotations = load_annotations()
res1 = []
res2 = []
with open('res1.json') as f:
    res1 = json.load(f)
with open('res2.json') as f:
    res2 = json.load(f)
# save_




lines = []
lines.append('idx\timgs\terr1\terr2\tres1\tres2\n')
for idx in range(len(res1)):
    err1 = (res1[idx]['label'] - res1[idx]['prediction'])
    err2 = (res2[idx]['label'] - res2[idx]['prediction'])
    if err1 == err2:
        continue
    line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(idx + 1, annotations[idx]['idx'], err1, err2, res1[idx], res2[idx])
    lines.append(line)
with open('diff.csv', 'w') as f:
    f.writelines(lines)
print("There are {} diff".format(len(lines) - 1))


lines = []
lines.append('idx\timgs\terr1\terr2\tres1\tres2\n')
for idx in range(len(res1)):
    err1 = (res1[idx]['label'] - res1[idx]['prediction'])
    err2 = (res2[idx]['label'] - res2[idx]['prediction'])
    if err1 == err2 or err1 != 0:
        continue
    line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(idx + 1, annotations[idx]['idx'], err1, err2, res1[idx], res2[idx])
    lines.append(line)


with open('diff_pos.csv', 'w') as f:
    f.writelines(lines)

print("There are {} pos diff".format(len(lines) - 1))

lines = []
lines.append('idx\timgs\terr1\terr2\tres1\tres2\n')
for idx in range(len(res1)):
    err1 = (res1[idx]['label'] - res1[idx]['prediction'])
    err2 = (res2[idx]['label'] - res2[idx]['prediction'])
    if err1 == err2 or err1 == 0:
        continue
    line = "{}\t{}\t{}\t{}\t{}\t{}\n".format(idx + 1, annotations[idx]['idx'], err1, err2, res1[idx], res2[idx])
    lines.append(line)
with open('diff_neg.csv', 'w') as f:
    f.writelines(lines)
print("There are {} neg diff".format(len(lines) - 1))
