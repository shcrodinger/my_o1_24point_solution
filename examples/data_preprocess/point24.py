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
def extract_solution(nums, target=24):
    
    def permutation(cnt, tmp):
        if cnt == 0:
            return next_step(deepcopy(tmp), [], True, [])
        else:
            res, i = [], 0
            while i < len(nums):
                tmp.append(nums.pop(i))
                res += permutation(cnt - 1, tmp)
                nums.insert(i, tmp.pop(-1))
                i += 1
            return res
    
    def next_step(ns, st, top0=False, path=[]):
        if len(st) == 1 and len(ns) == 0:
            return [(st[0], path[0])]
        res = []
        
        # 入栈
        if len(ns):
            num = ns.pop(0)
            st_bk = deepcopy(st) + [num]
            path_bk = deepcopy(path) + [str(num)]
            res += next_step(ns, st_bk, top0, path_bk)
            ns.insert(0, num)
        
        # 栈顶计算
        if len(st):
            if len(st) > 1:
                st_bk = deepcopy(st)
                path_bk = deepcopy(path)
                n2, n1 = st_bk.pop(-1), st_bk.pop(-1)
                p2, p1 = path_bk.pop(-1), path_bk.pop(-1)
                for sign in ('+', '-', '*', '/'):
                    if sign == '/' and n2 == 0: continue
                    st_bk_bk = deepcopy(st_bk)
                    path_bk_bk = deepcopy(path_bk)
                    cal_str = f'{n1}{sign}{n2}'
                    st_bk_bk.append(eval(cal_str))
                    path_bk_bk.append(f'({p1}{sign}{p2})')
                    res += next_step(ns, st_bk_bk, top0, path_bk_bk)
            elif len(st) == 1 and top0:
                res += next_step(ns, [-st[0]], False, ['-' + path[0]])
        
        return res

    answer = permutation(len(nums), [])
    res, info = 0, ''
    for ans in answer:
        if abs(ans[0] - target) < 1e-6:
            res, info = target, ans[1]
            break

    return res, info

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
            for n in range(1, 14):
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
            solution, reference = extract_solution(nums)
            
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
                    'reference': reference
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
