# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import time

import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
from easydict import EasyDict as edict

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import OFATokenizer, OFAModel

from volta.config import BertConfig
from volta.encoders import BertForVLTasks
from volta.train_utils import tbLogger
from volta.task_utils import LoadDatasetEval, LoadLoss, EvaluatingModel
from volta.datasets.from_dataset import get_dataset
tnz = OFATokenizer.from_pretrained("OFA-Sys/OFA-base")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/bert_config.json", type=str,
                        help="The config file which specified the model details.")
    # Output
    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--val_annotations_jsonpath", default="", type=str,
                        help="alternative annotations json path")
    parser.add_argument("--val_features_lmdbpath", default="", type=str,
                        help="alternative features lmdb path")
    # Text
    parser.add_argument("--do_lower_case", default=False, action="store_true",
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    # Evaluation
    parser.add_argument("--split", default="", type=str,
                        help="which split to use.")
    parser.add_argument("--batch_size", default=30, type=int,
                        help="batch size.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Seed
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    parser.add_argument("--use_chunk", default=0, type=float,
                        help="whether use chunck for parallel training.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Devices
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Output dirs
    timeStamp = args.from_pretrained.split("/")[-1] + "-" + args.save_name
    savePath = os.path.join(args.output_dir, timeStamp)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    save_path = "D:\\marvl-images\\zh\\images\\ofa_zh_test/"

    from datasets import load_from_disk
    def get_dataset_v1():
        s = load_from_disk(save_path)
        return s
    ds = get_dataset_v1()
    y = 0
    n = 0
    # for i in ds:
    #     if i["labels"][0] == 1:
    #         y += 1
    #     else:
    #         n += 1
    # print("get_dataset: {}, positives:{}, negatives:{}.".format((len(ds)), y, n))

    # Eval
    res_list = []
    model = OFAModel.from_pretrained("OFA-Sys/OFA-base", use_cache=False)
    # model.to(device)

    json_path = os.path.join(savePath, args.split, str(time.time()))
    with torch.no_grad():
        with open(json_path + "_result.json", "w") as f:
            for i in tqdm(range(len(ds))):
                picture = torch.tensor(ds[i]["picture"])
                tokens = torch.tensor([ds[i]["tokens"]])
                c = model.generate(tokens, patch_images=picture, num_beams=4)
                res = tnz.batch_decode(c, skip_special_tokens=True)
                d = ds[i]["labels"][0]
                f.write(str((res[0].strip(), d)) + "\n")
                res_list.append((res[0].strip(), d))
        json.dump(res_list, open(json_path + "_result1.json", "w"))

    # Metrics
    ans = []
    pred_pos = 0
    pred_false = 0
    for i in res_list:
        if i[0] == "yes":
            pred_pos += 1
            ans.append(i[1] == 1)
        else:
            pred_false += 1
            ans.append(i[1] == "False")
    print("total:{}, pred_pos:{},true_pos:{}, pred_false:{}, true_false:{}, acc:{}".format(len(ans), pred_pos, y,
                                                                                           pred_false, n,
                                                                                           sum(ans) / len(ans)))

    # results = []
    # others = []
    # for i, batch in tqdm(enumerate(dl_val), total=task2num_iters[task]):
    #     loss, score, batch_size, results, others = EvaluatingModel(config, task_cfg, device, task, batch,
    #                                                                model, dl_val, criterion, results, others)

    #     tb_logger.step_val(0, float(loss), float(score), task, batch_size, "val")
    #     sys.stdout.write("%d/%d\r" % (i, len(dl_val)))
    #     sys.stdout.flush()
    # # save the result or evaluate the result.
    # ave_score = tb_logger.showLossVal(task)
    # if task == "TASK12":
    #     from collections import defaultdict
    #     sent2corrects = defaultdict(list)
    #     for e in results:
    #         s = e["sentence"]
    #         # s1 = s[s.index('Question'):]
    #         sent2corrects[s].append(e["prediction"] == e["label"])
    #     s = 0
    #     for l in sent2corrects.values():
    #         s += (sum(l) == len(l))
    #     consistency = float(s) / len(sent2corrects) * 100
    #     logger.info(f"Consistency: {consistency}")

    # if args.split:
    #     json_path = os.path.join(savePath, args.split)
    # else:
    #     json_path = os.path.join(savePath, task_cfg[task]["val_split"])
    # json.dump(results, open(json_path + "_result.json", "w"))
    # json.dump(others, open(json_path + "_others.json", "w"))


if __name__ == "__main__":
    main()
