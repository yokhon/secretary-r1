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
    elif template_type == 't3':
        prefix = f"""Answer the given question. You must first conduct reasoning inside <think>...</think>. \
        If you find you lack some information or need further validation, you can raise a query by <query>...</query> after <think>...</think>. \
        Some questions may benefit from breaking down into several queries. A junior helper with skills (such as searching the Internet or using a calculator) will handle the query and return the result. \
        Once you have enough information, output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer> 13 </answer>. \
        \n \
        Output format for raise query: \
        <think>...</think><query>...</query> \
        \n \
        Output format for answer: \
        <think>...</think><answer>...</answer> \
        \n \
        Question: {question}\n"""
    elif template_type == 't4':
        prefix = f"""Answer the given question. You must first conduct reasoning inside <think>...</think>. \
        If you find you lack some information or need further validation, you can raise a query by <query>...</query> after <think>...</think>. \
        Some questions may benefit from breaking down into several queries. A junior helper with tools (such as search engine or calculator) will handle the query and return the result. \
        When you have the final answer, output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer> 13 </answer>. \
        \n \
        Output format for raise query: \
        <think>...</think><query>...</query> \
        \n \
        Output format for answer: \
        <think>...</think><answer>...</answer> \
        \n \
        Question: {question}\n"""
    elif template_type == 'swirl':
        prefix = f"""Please help me answer the following question in just a few words. \
        If you think it would help to use a calculator, please generate a mathematical query enclosed by <query>MATH EXP</query> tags. \
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        I will provide results in the following format: <info>RESULT</info>. \
        Once you have enough information, generate an answer enclosed by <answer>ANSWER</answer> tags. \
        Please either issue a calculation query or answer the question, but not both. \
        The question is: {question}\n"""
    elif template_type == 'swirl-v2':
        prefix = f"""Please help me answer the following question in just a few words. \
        If you think it would help to use a calculator, please generate a mathematical query enclosed by <math_exp>...</math_exp> tags. \
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        I will provide results in the following format: <info>...</info>. \
        Once you have enough information, generate an answer enclosed by <answer>...</answer> tags. \
        Please either issue a calculation query or answer the question, but not both. \
        The question is: {question}\n"""
    elif template_type == 'swirl-v3':
        prefix = f"""Answer the given question. You must first conduct reasoning inside <think>...</think>. \
        If you think it would help to use a calculator, please raise a mathematical query by <math_exp>...</math_exp> after <think>...</think>. \
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer> 13 </answer>. \
        The question is: {question}\n"""
    elif template_type == 'swirl-v4':
        prefix = f"""Answer the given question. If you think it would help to use a calculator, please raise a mathematical query by <math_exp>...</math_exp>. \
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer> 13 </answer>. \
        The question is: {question}\n"""
    elif template_type == 'swirl-v5':
        prefix = f"""Answer the given question. You must first conduct reasoning inside <think>...</think>. \
        If you think it would help to use a calculator, please raise a mathematical query by <math_exp>...</math_exp> after <think>...</think>. Then the calculation result will be provided in <info>...</info> tags.\
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here is an example: \
        ### Example Start ### \
        Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \
        Answer: <think>Natalia sold 48/2 clips in May.</think><math_exp>48/2</math_exp><info>The result is 24.</info><think>Natalia sold 48+24 clips altogether in April and May.</think><math_exp>48+24</math_exp><info>The result is 72.</info><think>I have the answer.</think><answer>72</answer> \
        ### Example End ### \
        Now, please answer the following question as per the specified format: {question}\n"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='./data/gsm8k')
    parser.add_argument('--local_dir', default='./data/my_gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='t2')
    parser.add_argument('--numbers', type=int, default=-1)

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset("parquet",
                                    data_files={
                                        'train': os.path.join(args.source_dir, 'train.parquet'),
                                        'test': os.path.join(args.source_dir, 'test.parquet')
                                    })

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

    train_dataset = dataset['train']
    if args.numbers > 0:
        test_dataset = dataset['train'].select(list(range(args.numbers)))
    else:
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