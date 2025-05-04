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
    elif template_type == 't2-1shot':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here is an example: \
        ### Example Start ### \
        Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \
        Answer: <think>Natalia sold 48/2 clips in May.</think><query>48/2</query><info>The result is 24.</info><think>Natalia sold 48+24 clips altogether in April and May.</think><query>48+24</query><info>The result is 72.</info><think>I have the answer.</think><answer>72</answer> \
        ### Example End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-1shot-v2':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here is an example: \
        ### Example Start ### \
        Question: Jason is planning a parking garage that will have 12 floors. Every 3rd floor has a gate where drivers have to show ID, which takes two minutes. To get from one floor to the next, drivers have to drive 800 feet at 10 feet/second. How long, in seconds, does it take to get to the bottom of the garage from the top? \
        Answer: <think>First, calculate the total number of gates someone has to pass through: 12 floors / 3 floors per gate.</think><query>Calculate 12/3</query><info>The result is 4.</info><think>Next, calculate the total time spent waiting at the gates: 4 gates * 2 minutes per gate.</think><query>Calculate 4*2</query><info>The result is 8.</info><think>Convert the waiting time from minutes to seconds: 8 minutes * 60 seconds per minute.</think><query>Calculate 8*60</query><info>The result is 480.</info><think>Calculate the time to drive through one floor in seconds: 800 feet / 10 feet per second.</think><query>Calculate 800/10</query><info>The result is 80.</info><think>Multiply the time per floor by the number of floors to find the total driving time: 80 seconds per floor * 12 floors.</think><query>Calculate 80*12</query><info>The result is 960.</info><think>Add the total driving time to the total time waiting at the gates to find the total time to get to the bottom: 960 seconds + 480 seconds.</think><query>Calculate 960+480</query><info>The result is 1440.</info><think>I have the answer.</think><answer>1440</answer> \
        ### Example End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-1shot-v3':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here is an example: \
        ### Example Start ### \
        Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \
        Answer: <think>Natalia sold 48/2 clips in May.</think><query>Calculate 48/2</query><info>The result is 24.</info><think>Natalia sold 48+24 clips altogether in April and May.</think><query>Calculate 48+24</query><info>The result is 72.</info><think>I have the answer.</think><answer>72</answer> \
        ### Example End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-1shot-v4':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here is an example: \
        ### Example Start ### \
        Question: Jane, Kyla, and Anthony have summer jobs in a resort. Their task is to fold guests' towels. Jane can fold 3 towels in 5 minutes. Kyla can fold 5 towels in 10 minutes, and Anthony can fold 7 towels in 20 minutes. If they all fold towels together, how many towels can they fold in one hour? \
        Answer: <think>There are 1 x 60 minutes in 1 hour.</think><query>Calculate 1*60</query><info>The result is 60.</info><think>There are 60/5 sets of 5 minutes in 1 hour.</think><query>Calculate 60/5</query><info>The result is 12.</info><think>Jane can fold 3 x 12 towels in an hour.</think><query>Calculate 3*12</query><info>The result is 36.</info><think>There are 60/10 sets of 10 minutes in 1 hour.</think><query>Calculate 60/10</query><info>The result is 6.</info><think>Kyla can fold 5 x 6 towels in an hour.</think><query>Calculate 5*6</query><info>The result is 30.</info><think>There are 60/20 sets of 20 minutes in 1 hour.</think><query>Calculate 60/20</query><info>The result is 3.</info><think>Anthony can fold 7 x 3 towels in an hour.</think><query>Calculate 7*3</query><info>The result is 21.</info><think>The 3 of them can fold a total of 36 + 30 + 21 towels in 1 hour.</think><query>Calculate 36+30+21</query><info>The result is 87.</info><think>I have the answer.</think><answer>87</answer> \
        ### Example End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-2shot':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here are 2 examples: \
        ### Example 1 Start ### \
        Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \
        Answer: <think>Natalia sold 48/2 clips in May.</think><query>48/2</query><info>The result is 24.</info><think>Natalia sold 48+24 clips altogether in April and May.</think><query>48+24</query><info>The result is 72.</info><think>I have the answer.</think><answer>72</answer> \
        ### Example 1 End ### \
        ### Example 2 Start ### \
        Question: Jason is planning a parking garage that will have 12 floors. Every 3rd floor has a gate where drivers have to show ID, which takes two minutes. To get from one floor to the next, drivers have to drive 800 feet at 10 feet/second. How long, in seconds, does it take to get to the bottom of the garage from the top? \
        Answer: <think>First, calculate the total number of gates someone has to pass through: 12 floors / 3 floors per gate.</think><query>12/3</query><info>The result is 4.</info><think>Next, calculate the total time spent waiting at the gates: 4 gates * 2 minutes per gate.</think><query>4*2</query><info>The result is 8.</info><think>Convert the waiting time from minutes to seconds: 8 minutes * 60 seconds per minute.</think><query>8*60</query><info>The result is 480.</info><think>Calculate the time to drive through one floor in seconds: 800 feet / 10 feet per second.</think><query>800/10</query><info>The result is 80.</info><think>Multiply the time per floor by the number of floors to find the total driving time: 80 seconds per floor * 12 floors.</think><query>80*12</query><info>The result is 960.</info><think>Add the total driving time to the total time waiting at the gates to find the total time to get to the bottom: 960 seconds + 480 seconds.</think><query>960+480</query><info>The result is 1440.</info><think>I have the answer.</think><answer>1440</answer> \
        ### Example 2 End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-2shot-v2':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here are 2 examples: \
        ### Example 1 Start ### \
        Question: Jason is planning a parking garage that will have 12 floors. Every 3rd floor has a gate where drivers have to show ID, which takes two minutes. To get from one floor to the next, drivers have to drive 800 feet at 10 feet/second. How long, in seconds, does it take to get to the bottom of the garage from the top? \
        Answer: <think>First, calculate the total number of gates someone has to pass through: 12 floors / 3 floors per gate.</think><query>Calculate 12/3</query><info>The result is 4.</info><think>Next, calculate the total time spent waiting at the gates: 4 gates * 2 minutes per gate.</think><query>Calculate 4*2</query><info>The result is 8.</info><think>Convert the waiting time from minutes to seconds: 8 minutes * 60 seconds per minute.</think><query>Calculate 8*60</query><info>The result is 480.</info><think>Calculate the time to drive through one floor in seconds: 800 feet / 10 feet per second.</think><query>Calculate 800/10</query><info>The result is 80.</info><think>Multiply the time per floor by the number of floors to find the total driving time: 80 seconds per floor * 12 floors.</think><query>Calculate 80*12</query><info>The result is 960.</info><think>Add the total driving time to the total time waiting at the gates to find the total time to get to the bottom: 960 seconds + 480 seconds.</think><query>Calculate 960+480</query><info>The result is 1440.</info><think>I have the answer.</think><answer>1440</answer> \
        ### Example 1 End ### \
        ### Example 2 Start ### \
        Question: Jim collects model cars, and he has 301 models total. Jim has 4 times as many Buicks as Fords, and 3 more than twice the number of Fords than Chevys. How many Buicks does Jim have? \
        Answer: <think>Let x represent the number of Chevys. The number of Fords is 3+2x, and the number of Buicks is 4 times the number of Fords, which is 4(3+2x).</think><query>Solve x + (3+2x) + 4(3+2x) = 301</query><info>The result is x = 26.</info><think>Using x = 26, the number of Buicks is calculated as 12 + 8x.</think><query>Calculate 12 + 8*26</query><info>The result is 220.</info><think>I have the answer.</think><answer>220</answer> \
        ### Example 2 End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
    elif template_type == 't2-3shot':
        prefix = f"""Answer the given question. You can raise query to answer the question. A junior helper with skills (such as using search engine or calculator) will handle the query and return the result. You can raise queries as many times as you want. \
        You must first conduct reasoning inside <think>...</think>. If you find you lack some information, you can raise a query by <query>...</query> after <think>...</think>. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. Here are 3 examples: \
        ### Example 1 Start ### \
        Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? \
        Answer: <think>Natalia sold 48/2 clips in May.</think><query>48/2</query><info>The result is 24.</info><think>Natalia sold 48+24 clips altogether in April and May.</think><query>48+24</query><info>The result is 72.</info><think>I have the answer.</think><answer>72</answer> \
        ### Example 1 End ### \
        ### Example 2 Start ### \
        Question: Jason is planning a parking garage that will have 12 floors. Every 3rd floor has a gate where drivers have to show ID, which takes two minutes. To get from one floor to the next, drivers have to drive 800 feet at 10 feet/second. How long, in seconds, does it take to get to the bottom of the garage from the top? \
        Answer: <think>First, calculate the total number of gates someone has to pass through: 12 floors / 3 floors per gate.</think><query>12/3</query><info>The result is 4.</info><think>Next, calculate the total time spent waiting at the gates: 4 gates * 2 minutes per gate.</think><query>4*2</query><info>The result is 8.</info><think>Convert the waiting time from minutes to seconds: 8 minutes * 60 seconds per minute.</think><query>8*60</query><info>The result is 480.</info><think>Calculate the time to drive through one floor in seconds: 800 feet / 10 feet per second.</think><query>800/10</query><info>The result is 80.</info><think>Multiply the time per floor by the number of floors to find the total driving time: 80 seconds per floor * 12 floors.</think><query>80*12</query><info>The result is 960.</info><think>Add the total driving time to the total time waiting at the gates to find the total time to get to the bottom: 960 seconds + 480 seconds.</think><query>960+480</query><info>The result is 1440.</info><think>I have the answer.</think><answer>1440</answer> \
        ### Example 2 End ### \
        ### Example 3 Start ### \
        Question: Jane, Kyla, and Anthony have summer jobs in a resort. Their task is to fold guests' towels. Jane can fold 3 towels in 5 minutes. Kyla can fold 5 towels in 10 minutes, and Anthony can fold 7 towels in 20 minutes. If they all fold towels together, how many towels can they fold in one hour? \
        Answer: <think>There are 1 x 60 minutes in 1 hour.</think><query>1*60</query><info>The result is 60.</info><think>There are 60/5 sets of 5 minutes in 1 hour.</think><query>60/5</query><info>The result is 12.</info><think>Jane can fold 3 x 12 towels in an hour.</think><query>3*12</query><info>The result is 36.</info><think>There are 60/10 sets of 10 minutes in 1 hour.</think><query>60/10</query><info>The result is 6.</info><think>Kyla can fold 5 x 6 towels in an hour.</think><query>5*6</query><info>The result is 30.</info><think>There are 60/20 sets of 20 minutes in 1 hour.</think><query>60/20</query><info>The result is 3.</info><think>Anthony can fold 7 x 3 towels in an hour.</think><query>7*3</query><info>The result is 21.</info><think>The 3 of them can fold a total of 36 + 30 + 21 towels in 1 hour.</think><query>36+30+21</query><info>The result is 87.</info><think>I have the answer.</think><answer>87</answer> \
        ### Example 3 End ### \
        Now, please answer the following question as per the specified steps and format: {question}\n"""
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
    elif template_type == 'swirl-v6':
        prefix = f"""Answer the given question. You must first conduct reasoning inside <think>...</think>. \
        If you think it would help to use a calculator, please raise a mathematical query by <query>...</query> after <think>...</think>. \
        Some questions may benefit from using a calculator multiple times in order to answer, so I will allow you to make up to 10 sequential queries before answering the question. \
        Please do not repeat queries you have already issued, as this is a waste of time. \
        When you have the final answer, you can output the answer inside <answer>...</answer>, without detailed illustrations. For example, <answer>13</answer>. \
        The question is: {question}\n"""
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