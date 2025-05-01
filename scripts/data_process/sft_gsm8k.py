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


def make_prefix(question, template_type):
    if template_type == 't5':
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
    parser.add_argument('--source_jsonl', default='./data/handled_answers.jsonl')
    parser.add_argument('--local_dir', default='./data/sft_gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='t5')

    args = parser.parse_args()

    df = pd.read_json(args.source_jsonl, lines=True)
    dataset = datasets.Dataset.from_pandas(df)

    data_source = 'openai/gsm8k'

    dataset = dataset.train_test_split(test_size=0.1)

    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['question']
            answer = example['formatted_answer']

            processed_question = make_prefix(question, template_type=args.template_type)

            data = {
                "question": processed_question,
                "formatted_answer": answer,
            }
            return data

        return process_fn

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
