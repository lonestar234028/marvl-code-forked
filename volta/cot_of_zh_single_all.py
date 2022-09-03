import torch

from volta.datasets.form_dataset_ofa import get_dataset_v1
from volta.datasets.form_dataset_ofa import get_dataset
from transformers import OFATokenizer, OFAModel
import  sys
tnz = OFATokenizer.from_pretrained("OFA-Sys/OFA-large")
model = OFAModel.from_pretrained("OFA-Sys/OFA-large", use_cache=False)
from flair.data import Sentence
from flair.models import SequenceTagger



from torchvision.utils import save_image
import  json
def write_json(imgid, prompt):
    load_dict = {}
    with open('img2prompt0821.json', 'r') as f:
        load_dict = json.load(f)
        num_item = len(load_dict)

    print("imgid : {}, prompt:{}".format(imgid, prompt))

    load_dict[imgid] = prompt
    with open('img2prompt0821.json', 'w', encoding='utf-8') as f2:
        json.dump(load_dict, f2, ensure_ascii=False)

def write_json_only(load_dict):

    with open('img2prompt0821.json', 'w', encoding='utf-8') as f2:
        json.dump(load_dict, f2, ensure_ascii=False)

def mapping(key):
    if key.strip() ==  'yes':
        return 'many'
    elif key.strip() == 'no':
        return '0'
    else:
        return  key

def get_result(i, ds):

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
    prompt = "What is the scenario?"
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
    # print(txt)
    tokens = torch.tensor([data["tokens"]])
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
    txt1 = "Left part scenario:" + a1 + " ,right part scenario: " + a2 + '. Therefore, the sentence is right or wrong: ' + \
           data['sentence']
    # inputs = tnz([txt1], max_length=1024, return_tensors="pt")["input_ids"]
    # c = model.generate(inputs, patch_images=picture, num_beams=4)
    # res = tnz.batch_decode(c, skip_special_tokens=True)
    #
    # print(res)
    # print('sentence:' + data['sentence'])
    # print('sentence1:' + txt)
    # print('sentence2:' + txt1)
    # write_json(data['image_id_0'] + '##' + data['image_id_1'], txt1)
    # print('labels:' + str(data['labels']))
    # print('image_id_0:' + data['image_id_0'])
    # print('image_id_1:' + data['image_id_1'])
    #
    dictionary = {}
    # dictionary["image_id_0"] = data["image_id_0"]
    # dictionary["image_id_1"] = data["image_id_1"]
    # dictionary["question_id"] = data["question_id"]
    #
    # dictionary["sentence"] = data["sentence"]
    # dictionary["labels"] = data["labels"]
    # dictionary["chapter"] = data["chapter"]
    # dictionary["pred"] = res[0].strip()
    return data['image_id_0'] + '##' + data['image_id_1'], txt1

def main():
    ds = get_dataset_v1()
    res = {}

    for i in range(ds.num_rows):
        k,v = get_result(i, ds)
        print("imgid : {}, prompt:{}".format(k, v))
        res[k] = v
    write_json_only(res)
    # load tagger
    # tagger = SequenceTagger.load("flair/pos-english")




if __name__ == "__main__":
    main()
