# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import jsonlines
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_id_0": item["image_id_0"],
        "image_id_1": item["image_id_1"],
        "sentence": item["sentence"],
        "answer": item,
    }
    return entry

pic_path = "D:\\marvl-images\\zh\\images/"

def load_images_path():
    from datasets import load_from_disk
    save_path = "D:\\marvl-images\\zh\\images\\ofa_zh_test/"
    s = load_from_disk(save_path)
    ss = {}

    def makess(r):
        ss[r['image_id_0'] + '##' + r['image_id_1']] = r['picture']

    s.map(lambda r: makess(r))

    return ss

def _load_dataset(annotations_path):
    """Load entries
    """
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
            dictionary["scores"] = [1.0]
            items.append(dictionary)
            count += 1
            if count < 2:
                print("loading_annotations: ")
                print(dictionary)

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class MaRVLDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=37,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 2
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._max_seq_length = max_seq_length
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        self.entries = _load_dataset(annotations_jsonpath)
        self.tokenize(max_seq_length)
        self.tensorize()
        self.img_features = load_images_path()

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer(entry['sentence']).input_ids
            # tokens = tokens[:max_length - 2]
            # tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id_0 = entry["image_id_0"]
        image_id_1 = entry["image_id_1"]
        question_id = entry["question_id"]

        mix_num_boxes = min(self._max_region_num * 2)
        mix_boxes_pad = np.zeros((self._max_region_num * 2, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num * 2, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num * 2:
            image_mask.append(0)

        img_segment_ids = np.zeros((mix_features_pad.shape[0]))


        features = self.img_features[image_id_0 + '##' + image_id_1]
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        target = torch.zeros(self.num_labels)

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, question_id

    def __len__(self):
        return len(self.entries)
