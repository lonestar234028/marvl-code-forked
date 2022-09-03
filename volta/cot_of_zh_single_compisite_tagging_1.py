import torch

from volta.datasets.form_dataset_ofa import get_dataset_v1
from get_obj import read_json
from volta.datasets.form_dataset_ofa import get_dataset
from transformers import OFATokenizer, OFAModel
import  sys
tnz = OFATokenizer.from_pretrained("OFA-Sys/OFA-large")
model = OFAModel.from_pretrained("OFA-Sys/OFA-large", use_cache=False)




from torchvision.utils import save_image
import json
def write_json_only(load_dict):

    with open('img2prompt0825.json', 'w', encoding='utf-8') as f2:
        json.dump(load_dict, f2, ensure_ascii=False)


def get_result(i, ds, pos_taggins):

    data = ds[i]
    txt = data['sentence']
    # img = data['picture']
    p1 = torch.tensor(data['p1'])
    p2 = torch.tensor(data['p2'])
    imgid2promt = []

    #
    # # save_image(torch.tensor(img), 'bin_img.jpg')
    # # save_image(p1, 'bin_img_1.jpg')
    # # save_image(p2, 'bin_img_2.jpg')
    # # # make example sentence
    # sentence = Sentence(txt)
    # # # load tagger
    # tagger = SequenceTagger.load("flair/pos-english")
    # # predict NER tags
    # tagger.predict(sentence)
    # # print sentence
    # print(sentence)
    taggings = {}
    prompt = "Where is/are " + pos_taggins[i] + "?"
    #
    # for labels in sentence.labels:
    #     dict_label = (labels.shortstring).split('/')
    #     if len(dict_label) == 2:
    #         if dict_label[1].__contains__('NN') or dict_label[1].__contains__('NNS'):
    #             if not(dict_label[0] .__contains__('picture')) and \
    #                     not (dict_label[0].__contains__('left')) and \
    #                     not (dict_label[0].__contains__('right')) and\
    #                     not(dict_label[0] .__contains__('image')):
    #                 # taggings[dict_label[1]] = taggings.get(dict_label[1], '##') + dict_label[0] + '##'
    #                 prompt += dict_label[0][1:-1] + ' and '
    #                 taggings[dict_label[0]] = taggings.get(dict_label[0], 0) + 1
    # prompt = prompt[:len(prompt) - len(' and ')]
    # print(taggings)
    max = 0
    nn = ""

    for k,v in taggings.items():
        if v > max :
            max = v
            nn = k
    txt = "How many " + prompt[:len(prompt) - len(' and ')] + "?"
    txt = prompt
    # txt = "For " + nn
    print(txt)
    inputs = tnz([txt], max_length=1024, return_tensors="pt")["input_ids"]
    #
    picture = torch.tensor(data["picture"])
    c = model.generate(inputs, patch_images=p1, num_beams=4)
    res = tnz.batch_decode(c, skip_special_tokens=True)
    # print(res)
    a1 = res[0].strip()
    c = model.generate(inputs, patch_images=p2, num_beams=4)
    res = tnz.batch_decode(c, skip_special_tokens=True)
    # print(res)
    a2 = res[0].strip()
    txt1 =  pos_taggins[i] + ' is ' + a1.strip() + " on the left and " + pos_taggins[i] + ' is ' + a2.strip() + ' on the right. Therefore, the sentence is right or wrong: ' + \
           data['sentence']

    return data['image_id_0'] + '##' + data['image_id_1'], txt1, prompt

def main():
    ds = get_dataset_v1()
    js = read_json()
    print(js[18:30])
    res = {}

    for i in range(2,3):
        k,v,p = get_result(i, ds, js)
        print("imgid : {}, question:{}, prompt:{}".format(k, v, p))
        res[k] = v
    # write_json_only(res)




if __name__ == "__main__":
    main()
