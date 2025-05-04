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
Preprocess the QA dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    question = dp['question']

    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'cot':
        prefix = f"""Answer the given question. \
        You must conduct step by step reasoning before getting to the answer. \
        After reasoning, please provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> 13 </answer>. Question: {question}\n"""
    elif template_type == 'base':
        prefix = f"""Answer the given question. \
        Please provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> 13 </answer>. Question: {question}\n"""
    elif template_type == 't1':
        prefix = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some information, you can raise a query by <query> sub-question </query>. \
        A junior helper with skills (such as searching the Internet or coding) will handle the query and return the result between <information> and </information>. \
        You can raise queries as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    elif template_type == 't5':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as searching the Internet or using a calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information or need further validation, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer>13</answer>. \
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
    parser.add_argument('--local_dir', default='./data/qa')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='t5')
    parser.add_argument('--data_sources', default='nq')

    args = parser.parse_args()

    data_sources = args.data_sources.split(',')
    all_dataset = []

    for data_source in data_sources:

        if data_source != 'strategyqa':
            dataset = datasets.load_dataset('RUC-NLPIR/FlashRAG_datasets', data_source)
        else:
            dataset = datasets.load_dataset('json', data_files="/home/peterjin/mnt/data/strategyqa/test_correct.jsonl")

        if 'test' in dataset:
            print(f'Using the {data_source} test dataset...')
            test_dataset = dataset['test']
        elif 'dev' in dataset:
            print(f'Using the {data_source} dev dataset...')
            test_dataset = dataset['dev']
        else:
            print(f'Using the {data_source} train dataset...')
            test_dataset = dataset['train']


        # add a row to each data item that represents a unique id
        def make_map_fn(split):

            def process_fn(example, idx):
                example['question'] = example['question'].strip()
                if example['question'][-1] != '?':
                    example['question'] += '?'
                question = make_prefix(example, template_type=args.template_type)
                solution = {
                    "target": example['golden_answers'],
                }

                data = {
                    "data_source": data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "fact-reasoning",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": solution
                    },
                    "extra_info": {
                        'split': split,
                        'index': idx,
                    }
                }
                return data

            return process_fn


        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
        all_dataset.append(test_dataset)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    all_test_dataset = datasets.concatenate_datasets(all_dataset)
    all_test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)