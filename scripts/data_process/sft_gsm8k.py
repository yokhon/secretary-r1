# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the gsm8k SFT dataset to parquet format
"""

import re
import os
import datasets

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_jsonl', default='./data/handled_answers.jsonl')
    parser.add_argument('--local_dir', default='./data/sft_gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    df = pd.read_json(args.source_jsonl, lines=True)
    dataset = datasets.Dataset.from_pandas(df)

    data_source = 'openai/gsm8k'

    dataset = dataset.train_test_split(test_size=0.1)

    # def make_map_fn(split):

    #     def process_fn(example, idx):
    #         question = example['question']
    #         answer = example['formatted_answer']

    #         data = {
    #             "data_source": data_source,
    #             "extra_info": {
    #                 'split': split,
    #                 'index': idx,
    #                 'question': question,
    #                 'answer': answer,
    #             }
    #         }
    #         return data

    #     return process_fn

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
