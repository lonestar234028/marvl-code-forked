# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer

from volta.datasets.marvl_dataset_ofa_improved import MaRVLDataset
from volta.datasets._image_features_reader import ImageFeaturesH5Reader


logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="mean"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
}


def LoadDatasetEval(args, config, task_cfg, task_id):
    ckpt_dir = "OFA-Sys/OFA-base"
    from transformers import OFATokenizer
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)    # if "roberta" in args.bert_model:
    #     tokenizer = RobertaTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    # else:
    #     tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    task = "TASK" + task_id
    task_name = task_cfg[task]["name"]

    batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)
    if args.local_rank != -1:
        batch_size = int(batch_size / dist.get_world_size())

    logger.info("Loading %s Dataset with batch size %d" % (task_name, batch_size))
    if args.split:
        eval_split = args.split
    else:
        eval_split = task_cfg[task]["val_split"]

    val_annotations_jsonpath = args.val_annotations_jsonpath or task_cfg[task]["val_annotations_jsonpath"]

    dset_val = MaRVLDataset(
        task=task_cfg[task]["name"],
        dataroot=task_cfg[task]["dataroot"],
        annotations_jsonpath=val_annotations_jsonpath,
        split=eval_split,
        tokenizer=tokenizer,
        bert_model=args.bert_model,
        padding_index=0,
        max_seq_length=task_cfg[task]["max_seq_length"],
        max_region_num=task_cfg[task]["max_region_num"],
        num_locs=config.num_locs,
        add_global_imgfeat=config.add_global_imgfeat,
        append_mask_sep=(config.fusion_method == 'vl-bert_vqa'),
    )

    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=args.drop_last,
    )
    task2num_iters = {task: len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val
