# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Classification fine-tuning: utilities to work with various datasets """

from __future__ import absolute_import, division, print_function
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import csv

SUPPORTED_DATASETS = ["sbic","stereoset","all"]
DATASET_TO_PATH = {"sbic": {"train":"SBIC/SBIC.v2.agg.trn.csv", 
                            "dev": "SBIC/SBIC.v2.agg.dev.csv",
                            "test": "SBIC/SBIC.v2.agg.tst.csv"},
                    "stereoset": {"train":"stereoset_train.csv",
                                  "dev":"stereoset_dev.csv",
                                  "test":"stereoset_test.csv"},
                    "all": {"train":"all_train.csv",
                                  "dev":"all_dev.csv",
                                  "test":"all_test.csv"}}

DATASET_TO_LABEL_COLNAME = {"sbic": {"train": "hasBiasedImplication",
                                       "dev": "hasBiasedImplication",
                                       "test": "hasBiasedImplication"},
                            "stereoset": {"train":"label",
                                        "dev":"label",
                                        "test":"label"},
                            "all": {"train":"label",
                                    "dev":"label",
                                    "test":"label"}}

class SbicDataset(IterableDataset):
    def __init__(self, split, tokenizer, batch_size, dataset_name, toxicity_threshold=0.5):
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError("Dataset not supported")
        self.path, self.target_colname = self.get_split_metadata(split, dataset_name)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.dataset_name = dataset_name

    def get_split_metadata(self, split, dataset_name):
        # if dataset_name == "all":
        #     return "", ""
        path = DATASET_TO_PATH[dataset_name][split]
        target_colname = DATASET_TO_LABEL_COLNAME[dataset_name][split]
        return path, target_colname

    def get_single_stream(self):
        """Creates examples for the training and dev sets from labelled folder."""
        MAX_LENGTH = 126
        with open(self.path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for data in reader:
                x_vals = data['sentence']
                ids = int(data["id"])
                y_vals = int(data[self.target_colname])
                tokenized = self.tokenizer([x_vals], truncation=True, padding="max_length", max_length=MAX_LENGTH, return_token_type_ids=True)
                tokenized["input_ids"] = torch.tensor(tokenized["input_ids"][0])
                tokenized["token_type_ids"] = torch.tensor(tokenized["token_type_ids"][0])
                tokenized["attention_mask"] = torch.tensor(tokenized["attention_mask"][0])
                tokenized["label"] = torch.tensor(y_vals)
                tokenized["id"] = torch.tensor(ids)
                yield tokenized

    def __iter__(self):
        return self.get_single_stream()

