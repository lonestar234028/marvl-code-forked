import torch

from volta.datasets.form_dataset_ofa import get_dataset_v1
from volta.datasets.form_dataset_ofa import get_dataset
from transformers import OFATokenizer, OFAModel
import sys
from flair.data import Sentence
from flair.models import SequenceTagger
import json
# import these modules
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ps = PorterStemmer()
#
# tagger = SequenceTagger.load("flair/pos-english")
from random import choice
def deal_x(x):
    if 'people' in x or 'person' in x:
        return 'people'
    x = x.strip().split('\t')
    pos = list(map(lambda y: y.split(':')[0], x[2][2:-1].replace('"','').replace("'",'').split(",")))

    return pos[-1]
    # print(json.loads(pos))

def read_json():
    lines = []
    with open('taggins.csv', 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    print(len(lines))
    return list(map(deal_x, lines))

def write_json_only(load_dict):
    with open('img2prompt0821.json', 'w', encoding='utf-8') as f2:
        json.dump(load_dict, f2, ensure_ascii=False)


def mapping(key):
    if key.strip() == 'yes':
        return 'many'
    elif key.strip() == 'no':
        return '0'
    else:
        return key


def get_result(i, ds):
    data = ds[i]
    txt = data['sentence']
    print(txt)
    # img = data['picture']
    # p1 = torch.tensor(data['p1'])
    # p2 = torch.tensor(data['p2'])
    # imgid2promt = []
    #
    # #
    # # # save_image(torch.tensor(img), 'bin_img.jpg')
    # # # save_image(p1, 'bin_img_1.jpg')
    # # # save_image(p2, 'bin_img_2.jpg')
    # # # # make example sentence
    # sentence = Sentence(txt)
    # # # load tagger
    #
    # # predict NER tags
    # tagger.predict(sentence)
    # # print sentence
    # # print(sentence)
    # taggings = {}
    # # prompt = "What is the scenario?"
    # #
    # for labels in sentence.labels:
    #     dict_label = (labels.shortstring).split('/')
    #     if len(dict_label) == 2:
    #         if dict_label[1].__contains__('NN') or dict_label[1].__contains__('NNS'):
    #             if not (dict_label[0].__contains__('picture')) and \
    #                     not (dict_label[0].__contains__('left')) and \
    #                     not (dict_label[0].__contains__('right')) and \
    #                     not (dict_label[0].__contains__('image')):
    #                 # taggings[dict_label[1]] = taggings.get(dict_label[1], '##') + dict_label[0] + '##'
    #                 taggings[dict_label[0]] = taggings.get(dict_label[0], 0) + 1
    #
    # # print(taggings)
    # # for w in words:
    # #     print(w, " : ", ps.stem(w))
    # max = 0
    # nn = ""
    # return  data['image_id_0'] + '##' + data['image_id_1'], taggings

def main():
    ds = get_dataset_v1()
    res = {}

    lines = []
    lines.append('idx\tk\tv\n')
    for i in range(ds.num_rows):
        get_result(i, ds)
        # line = "{}\t{}\t{}\n".format(i, k, v)
        # lines.append(line)
    # with open('taggins.csv', 'w') as f:
    #     f.writelines(lines)
    # write_json_only(res)
    # load tagger
    # tagger = SequenceTagger.load("flair/pos-english")


if __name__ == "__main__":
    main()
