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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import json
import random

from verl.utils.hdfs_io import copy, makedirs
import argparse
from copy import deepcopy
from tqdm import tqdm
from datasets import Dataset

# 构建24点
def extract_solution(nums):
    
    def permutation(cnt, tmp):
        if cnt == 0:
            return next_step(deepcopy(tmp), [])
        else:
            res, i = [], 0
            while i < len(nums):
                tmp.append(nums.pop(i))
                res += permutation(cnt - 1, tmp)
                nums.insert(i, tmp.pop(-1))
                i += 1
            return res
    
    def next_step(ns, st):
        if len(ns) == 0 and len(st) == 1:
            return [sum(st)]
        
        empty_ns = (len(ns) == 0)
        num = ns.pop(0) if not empty_ns else st.pop(-1)
        
        res = []
        
        # 结合
        for sign in ('+', '-', '*', '/'):
            if (sign == '/' and num == 0) or (len(st) == 0 and sign in ('*', '/')):
                continue
            st_bk = deepcopy(st)
            n = st_bk.pop(-1) if len(st) else 0
            st_bk.append(eval(f'{n}{sign}{num}'))
            res += next_step(ns, st_bk)
        
        # 入栈
        if not empty_ns:
            st.append(num)
            res += next_step(ns, st)
            st.pop(-1)
            ns.insert(0, num)
        else:
            st.append(num)

        return res
    
    answer = permutation(len(nums), [])

    return 24 if 24 in answer else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_num', default=4)
    parser.add_argument('--local_dir', default='~/data/point24')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '24point'
    
    # 测试集占比
    test_prob = 0.1
    train_data, test_data = [], []
    exist = set()
    def find_all_nums(depth, tmp):
        if depth:
            for n in range(10):
                tmp.append(n)
                find_all_nums(depth - 1, tmp)
                tmp.pop(-1)
        else:
            num_key = json.dumps(sorted(deepcopy(tmp)))
            if num_key in exist: return
            exist.add(num_key)
            if random.uniform(0, 1) < test_prob:
                test_data.append({'question': deepcopy(tmp)})
            else:
                train_data.append({'question': deepcopy(tmp)})
    
    find_all_nums(args.limit_num, [])
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    instruction_following = '''Calculate 24 points using {nums}. If there is a solution, please output a mathematical expression. If there is no solution, please output \"no solution\". You need to think about the reasoning process in the mind, reflect and verify them by yourself, and correct them until the problem is solved. And Just provide one final answer. Please think and output in the following format.
output format:
<think>
xxx (Here is your thinking process. For the calculation of 24 points, the thinking process is to keep changing the combination order of each number and the calculation method until the final answer is obtained.)
</think>
The answer is xxx (Here is your final answer.)'''
    # instruction_following = "Let's think step by step and output the final answer after \"####\"."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            nums = example.pop('question')
    
            question_raw = ','.join([str(n) for n in nums])
            question = instruction_following.format(nums=question_raw)
            solution = extract_solution(nums)
            
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
                    'answer': '',
                    "question": question_raw,
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
