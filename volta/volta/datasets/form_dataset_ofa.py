from PIL import Image, ImageFile
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
import torch
from datasets import Dataset
import pandas as pd
import jsonlines
import os
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 384
pic_path = "D:\\marvl-images\\zh\\images/"
annotations_path = "C:\\Users\\taoli1\code\\MultiModal\\ofa_proj\\marvl-code-forked\\data\\zh\\annotations_machine-translate\\marvl-zh_gmt.jsonl"
save_path = "D:\\marvl-images\\zh\\images\\ofa_zh_test_large/"
ckpt_dir = "OFA-Sys/OFA-large"

# tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

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

            dictionary["sentence"] = str(annotation["caption"])
            dictionary["labels"] = [int(annotation["label"])]
            dictionary["chapter"] = str(annotation["chapter"])
            dictionary["scores"] = [1.0]
            items.append(dictionary)
            count += 1
            if count < 2:
                print("loading_annotations: ")
                print(dictionary)
    return items


def load_images_path():
    paths = {}
    for dirs in os.listdir(pic_path):
        for dir in os.listdir(pic_path + dirs):
            path_dir = (pic_path + dirs + '/' + dir)
            paths[dir.split(".")[0]] = path_dir
    return paths


def merge_picture(pic1, pic2):
    patch_img1 = patch_resize_transform(pic1)
    patch_img2 = patch_resize_transform(pic2)
    patch_img = torch.cat((patch_img1, patch_img2), dim=-1)
    return patch_img.unsqueeze(0), patch_img1.unsqueeze(0), patch_img2.unsqueeze(0)


def insert_image(item, paths):
    pic1 = Image.open(paths[item['image_id_0']])
    pic2 = Image.open(paths[item['image_id_1']])
    tokens = tokenizer(item['sentence']).input_ids
    picture,p1,p2 = merge_picture(pic1, pic2)
    newitem = item
    newitem["picture"] = picture
    newitem["p1"] = p1
    newitem["p2"] = p2
    newitem["tokens"] = tokens
    return newitem


def get_dataset():
    items = load_annotations()
    paths = load_images_path()
    dataset = []
    n = 0
    for item in items:
        data = insert_image(item, paths)
        n += 1
        if n > 1:
            break
        dataset.append(data)
    return dataset

from datasets import load_from_disk

def get_dataset_v1():
    dataset = load_from_disk(save_path)
    return dataset
# data = load_annotations()
# data = pd.DataFrame(data)
# paths = load_images_path()
# data = Dataset.from_pandas(data)
# data = data.map(lambda d: insert_image(d, paths))
# print(data)
# data.save_to_disk(save_path)