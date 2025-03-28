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

import re
from copy import deepcopy

# 获取当前装字符串
def get_state_key(mt):
    m, n = len(mt), len(mt[0])
    state_key = '#'.join([str(mt[i][j]) for i in range(m) for j in range(n)])
    return state_key

def clean_solution_str(solution_str):
    solution_str = solution_str.lower().strip()
    solution_str = solution_str[re.search('assistant\n', solution_str).end():].strip()
    return solution_str
    
def regext_format_score(solution_str):
    format_ret = 0
    
    solution_str = clean_solution_str(solution_str)
    format_pattern = '^<think>[\s\S]+</think>[\s\S]+<answer>[\s\S]+<answer>$'
    if re.search(format_pattern, solution_str):
        format_ret += 0.5
    
    if solution_str.count('<think>') == 1:
        format_ret += 0.5 / 4
    
    if solution_str.count('</think>') == 1:
        format_ret += 0.5 / 4
    
    if solution_str.count('<answer>') == 1:
        format_ret += 0.5 / 4
    
    if solution_str.count('</answer>') == 1:
        format_ret += 0.5 / 4
    
    return format_ret

# puzzle move is valid
def check_answer_valid(mt, actions, reference, target, method='flexible'):
    mt_bk = deepcopy(mt)
    
    # 终结状态
    TERMINAL_STATE = get_state_key(target)

    m, n = len(mt_bk), len(mt_bk[0])
    
    for i in range(m):
        for j in range(n):
            if mt_bk[i][j] == 0:
                break
        if mt_bk[i][j] == 0: break
   
    ops = {
        'up': (1, 0),
        'down': (-1, 0),
        'left': (0, 1),
        'right': (0, -1)
    }
    
    ret = 0.0
    valid_action_cnt = 0

    for action in actions:
        opx, opy = ops.get(action, (0, 0))
        if 0 <= i + opx <= m - 1 and 0 <= j + opy <= n - 1:
            valid_action_cnt += 1
            mt_bk[i][j], mt_bk[i + opx][j + opy] = mt_bk[i + opx][j + opy], mt_bk[i][j]
            i, j = i + opx, j + opy
        else:
            # 越界没有收益
            return ret
    
    # 统计correctness
    if method == 'strict':
        if get_state_key(mt_bk) == TERMINAL_STATE:
            return 1.0
        else: return 0.0
    
    if get_state_key(mt_bk) == TERMINAL_STATE:
        ret = 1.0 - min(max(valid_action_cnt - len(reference), 0) * 0.1, 0.5)
    else:
        # 不越界，稍微奖励一下
        ret = 0.1
    
    return ret

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']
    
    solution_str = clean_solution_str(solution_str)

    if method == 'strict':
        # this also tests the formatting of the model
        ma = re.search("<answer>((left|right|up|down|\\-|\\s)+)</answer>", solution_str)
        if not ma:
            final_answer = []
        else:
            final_answer = ma.group(1).strip().split('-')
    elif method == 'flexible':
        ma = re.search("((left|right|up|down|\\-|\\s)+)", solution_str)
        final_answer = []
        if ma:
            print(ma.group())
            final_answer = ma.group(1).strip().split('-')

    return final_answer

def compute_score(solution_str, ground_truth, extra_info, method='strict', format_score=0.4, score=1.):
    """The scoring function for 24point.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    ndarry_to_list = lambda xs: [list(x) for x in xs]
    grid = ndarry_to_list(extra_info['grid'])
    target = ndarry_to_list(extra_info['target'])

    answer = extract_solution(solution_str=solution_str, method=method)
    regex_format_ret = regext_format_score(solution_str)
    
    reference = ground_truth.split('-')
    
    total_score = check_answer_valid(grid, answer, reference, target) * score
    total_score += regex_format_ret * format_score
    
    correctness = check_answer_valid(grid, answer, reference, target, 'strict')
    formatness = regex_format_ret * format_score
    
    return total_score, correctness, formatness