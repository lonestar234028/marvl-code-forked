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
save_path = "D:\\marvl-images\\zh\\images\\zh_test_images/"
ckpt_dir = "OFA-Sys/OFA-base"
import  shutil
def load_images_path():
    paths = {}
    for dirs in os.listdir(pic_path):
        for dir in os.listdir(pic_path + dirs):
            path_dir = (pic_path + dirs + '/' + dir)
            paths[dir.split(".")[0]] = path_dir
            shutil.copyfile(path_dir, save_path + dir)

    return paths


paths = load_images_path()

print(paths)