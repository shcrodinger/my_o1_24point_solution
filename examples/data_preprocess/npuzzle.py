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
import heapq
import pdb

from verl.utils.hdfs_io import copy, makedirs
import argparse
from copy import deepcopy
from tqdm import tqdm
from datasets import Dataset

# 获取当前装字符串
def get_state_key(mt):
    m, n = len(mt), len(mt[0])
    state_key = '#'.join([str(mt[i][j]) for i in range(m) for j in range(n)])
    return state_key

# a*算法
def a_star_explore(mt, terminal, step_limit=20):
    m, n = len(mt), len(mt[0])
    # 启发式
    def heuristic(mtbk):
        ret, ret_max = 0, 0
        for i in range(m):
            for j in range(n):
                if mtbk[i][j] == 0:
                    ret += abs(i - (m - 1)) + abs(j - (n - 1))
                    ret_max = max(ret_max, abs(i - (m - 1)) + abs(j - (n - 1)))
                else:
                    ri, rj = (mtbk[i][j] - 1) // n, (mtbk[i][j] - 1) % n
                    ret += abs(ri - i) + abs(rj - j)
                    ret_max = max(ret_max, abs(ri - i) + abs(rj - j))
        
        # ret = min(ret_max, ret // 9)

        return ret_max
    
    # 寻找0的位置
    x, y = 0, 0
    for i in range(m):
        for j in range(n):
            if mt[i][j] == 0:
                x, y = i, j
                break
    
    path = ''
    prior_que = [(0, x, y, mt, path)]   # cost x y map
    exist_state = {}
    # a*探索
    has_ret = False
    while len(prior_que):
        _, x, y, mt, path = heapq.heappop(prior_que)
        path_cost = len(path.split('-'))

        if get_state_key(mt) == terminal:
            has_ret = True
            break
        ops = []

        if x > 0: ops.append((-1, 0))
        if y > 0: ops.append((0, -1))
        if x < m - 1: ops.append((1, 0))
        if y < n - 1: ops.append((0, 1))
        for opx, opy in ops:
            mt_tmp = deepcopy(mt)
            mt_tmp[x][y], mt_tmp[x + opx][y + opy] = mt_tmp[x + opx][y + opy], mt_tmp[x][y]
            mt_key = get_state_key(mt_tmp)
            if mt_key not in exist_state or exist_state[mt_key] > path_cost:
                if mt_key == terminal: cost = len(path.split(',')) - 1
                else: cost = heuristic(mt_tmp)
                new_path = (path if path.strip() == '' else f'{path}-') + ('up' if opx == 1 else 'down' if opx == -1 else 'left' if opy == 1 else 'right')
                heapq.heappush(prior_que, (path_cost + cost, x + opx, y + opy, mt_tmp, new_path))
                exist_state[mt_key] = path_cost
    
    return path if has_ret else ''

# npuzzle数据
def npuzzle(n=3, step=20, dest_cnt=10000):
    # init map
    grid = [[(i * n + j + 1) for j in range(n)] for i in range(n)]
    grid[-1][-1] = 0

    terminal_state = get_state_key(grid)
    data = []
    exist_state = set()
    
    for _ in range(dest_cnt):
        real_step = random.randint(max(1, step // 2), step)
        grid_tmp = deepcopy(grid)
        x, y = n - 1, n - 1
        while real_step:
            opx, opy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            while not((opx, opy) != (0, 0) and -1 < x + opx < n and -1 < y + opy < n):
                opx, opy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            real_step -= 1
            grid_tmp[x][y], grid_tmp[x + opx][y + opy] = grid_tmp[x + opx][y + opy], grid_tmp[x][y]
            x, y = x + opx, y + opy
        
        grid_key = get_state_key(grid_tmp)
        if grid_key not in exist_state:
            path = a_star_explore(grid_tmp, terminal_state)
            data.append({
                'grid': grid_tmp,
                'reference': path,
            })
            exist_state.add(grid_key)

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=3)
    parser.add_argument('--local_dir', default='~/data/8puzzle')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '8puzzle'

    # 测试集占比
    test_prob = 0.1
    train_data, test_data = [], []
    exist_mt_key = set()
    
    # 生成8puzzle
    dest_cnt = 1000000
    data = npuzzle(3, step=10, dest_cnt=dest_cnt)
    random.shuffle(data)
    dest_cnt = len(data)

    train_num = int(dest_cnt * (1 - test_prob))
    train_data, test_data = data[:train_num], data[train_num:]
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    instruction_following = '''You're a master at playing the 8-puzzle game. 

Objective:
Your goal is to transform the initial state of the puzzle into the target state by sliding the tiles.
The target state is typically a configuration where the tiles are arranged in numerical order from 1 to 8, with the 0 located at the bottom-right corner. For example:
[1 2 3]
[4 5 6]
[7 8 0]

Rules of the Game:
Initial State: The 8 tiles and the empty space are randomly arranged in the 3x3 grid.
Movement Rules:
The 0 represents the empty space.
Only tiles that are adjacent to the 0 (up, down, left, or right) can be moved into the 0.
Tiles cannot move diagonally, and they cannot jump over other tiles.
Winning Condition:
The puzzle is solved when all tiles are arranged in numerical order, with the 0 in the bottom-right corner.

You need to think about the reasoning process in the mind, reflect and verify them by yourself, and correct them until the problem is solved.
The final answer consists of four types of operations: up, down, left, and right, which are connected by "-".
Here is a specific example for you.
example:
[1 2 3]
[4 5 6]
[0 7 8]
output format:
<think>
...(Here is your thought process. You can let your imagination run wild.)
</think>
<answer>
left-left(It means that moves 6 to the left into the empty position, and then move 7 to the left into the empty position.)
</answer>

Now, please move the following initial state to the target state.
Initial State:
{init_state}
'''
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            grid = example['grid']
            reference = example['reference']
            grid_str = ''
            for tmp in grid:
                grid_str += '[' + ' '.join([str(n) for n in tmp]) + ']\n'
            grid_str = grid_str.strip()
            question = instruction_following.format(init_state=grid_str)

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": reference
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'grid': grid
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