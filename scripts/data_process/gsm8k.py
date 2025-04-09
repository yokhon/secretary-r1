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
Preprocess the nq dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        prefix = f"""Answer the given question. \
                You must conduct step by step reasoning before getting to the answer. \
                After reasoning, please provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 't1':
        prefix = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some information, you can raise a query by <query> sub-question </query>. \
        A junior helper with skills (such as searching the Internet or coding) will handle the query and return the result between <information> and </information>. \
        You can raise queries as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 't2':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as searching the Internet or coding) will handle the query and return the result. You can raise queries as many times as you want. \
    You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
    When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. \
    \n \
    Output format for raise query: \
    <think>...</think><query>...</query> \
    \n \
    Output format for answer: \
    <think>...</think><answer>...</answer> \
    \n \
    Question: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='./data/gsm8k')
    parser.add_argument('--local_dir', default='./data/my_gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='t2')

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset("parquet",
                                    data_files={
                                        'train': os.path.join(args.source_dir, 'train.parquet'),
                                        'test': os.path.join(args.source_dir, 'test.parquet')
                                    })

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            example['question'] = example['extra_info']['question'].strip()
            if example['question'][-1] != '?':
                example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            solution = example['reward_model']['ground_truth']
            answer = example['extra_info']['answer']

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)