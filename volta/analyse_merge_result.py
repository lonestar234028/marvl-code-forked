import os
from collections import defaultdict
import json
import  sys

json_file_path = sys.argv[1]
jsons=[]
data = set()
for file in os.listdir(json_file_path):
    if file.endswith('_result.json'):
        continue
    with open(os.path.join(json_file_path, file)) as f:
        jsons.append(json.load(f))

print(len(jsons))
sent2corrects = defaultdict(list)
y_pred = 0
y = 0
n = 0
ans = []
pred_pos = 0
pred_false = 0
for jsonobj in jsons:
    for i in jsonobj:
        if i[1] == 1:
            y += 1
        else:
            n += 1
        if i[0] == "yes":
            pred_pos += 1
            ans.append(i[1] == 1)
            if i[1] == 1:
                y_pred += 1
        if i[0] == "no":
            pred_false += 1
            ans.append(i[1] == 0)
            if i[1] == 0:
                y_pred += 1

print("total:{}, pred_pos:{},true_pos:{}, pred_false:{}, true_false:{}, acc:{}".format(len(ans), pred_pos, y,
                                                                                           pred_false, n,
                                                                                           sum(y_pred) / len(ans)))
