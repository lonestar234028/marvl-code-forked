import torch

from volta.datasets.form_dataset_ofa import get_dataset_v1
from transformers import OFATokenizer, OFAModel
tnz = OFATokenizer.from_pretrained("OFA-Sys/OFA-base")
model = OFAModel.from_pretrained("OFA-Sys/OFA-base", use_cache=False)


def main():
    ds = get_dataset_v1()
    i = 0
    txt = "Firstly,"
    # tokens = torch.tensor([ds[i]["tokens"]])
    inputs = tnz([txt], max_length=1024, return_tensors="pt")["input_ids"]

    picture = torch.tensor(ds[i]["picture"])
    # c = model.generate(inputs, num_beams=4)
    c = model.generate(inputs, patch_images=picture, num_beams=4)
    res = tnz.batch_decode(c, skip_special_tokens=True)
    print("=========stage1=============")
    print(res)
    print("=========stage2=============")
    txt1 = txt + res[0] + '. Therefore, the sentence is right or wrong? (' + ds[i]['sentence'] + ')'
    inputs = tnz([txt1], max_length=1024, return_tensors="pt")["input_ids"]
    c = model.generate(inputs, patch_images=picture, num_beams=4)
    res = tnz.batch_decode(c, skip_special_tokens=True)

    print(res)
    print('sentence:' + ds[0]['sentence'])
    print('sentence1:' + txt)
    print('sentence2:' + txt1)
    print('labels:' + str(ds[0]['labels']))
    print('image_id_0:' + ds[i]['image_id_0'])
    print('image_id_1:' + ds[i]['image_id_1'])
if __name__ == "__main__":
    main()
