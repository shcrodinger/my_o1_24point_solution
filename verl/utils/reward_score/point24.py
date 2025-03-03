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

def clean_solution_str(solution_str):
    solution_str = solution_str.lower().strip()
    solution_str = solution_str[re.search('assistant\n', solution_str).end():].strip()
    solution_str = re.sub('\\\\times|×|x', '*', solution_str)
    solution_str = re.sub('\\\\div', '/', solution_str)
    solution_str = re.sub('\\\\', '', solution_str)
    
    return solution_str
    
def regext_format_score(solution_str):
    format_ret = 0
    
    solution_str = clean_solution_str(solution_str)
    format_pattern = '<think>[\s\S]+</think>[\s\S]+the answer is(\\s*no solution|[\\(\\)+\\-*/0-9\\s]+).{0,1}$'
    if re.search(format_pattern, solution_str):
        format_ret += 0.5
    
    if solution_str.count('<think>') == 1:
        format_ret += 0.5 / 3
    
    if solution_str.count('</think>') == 1:
        format_ret += 0.5 / 3
    
    if solution_str.count('the answer is') == 1:
        format_ret += 0.5 / 3
    
    return format_ret
    
        
def extract_solution(solution_str, extra_info, method='strict'):
    assert method in ['strict', 'flexible']
    
    solution_str = clean_solution_str(solution_str)

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.findall("the answer is\\s*no solution|the answer is[\\(\\)+\\-*/=0-9\\s]+", solution_str)
        if len(solution) == 0:
            final_answer = None
        else:
            final_answer = solution[-1]
            final_answer = re.sub('=\\s*24|24\\s*=', '', final_answer)
            final_answer = final_answer.split('the answer is')[1].strip()
            # print(solution_str)
    elif method == 'flexible':
        answer = re.findall("[\\(\\)+\\-*/=0-9\\s]+|no solution", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                final_answer = final_answer.strip()
                final_answer = re.sub('=\\s*24|24\\s*=', '', final_answer)
                try:
                    if final_answer == 'no solution':
                        break
                    eval(final_answer)
                    if final_answer not in invalid_str:
                        break
                except:
                    pass
    
    # print(final_answer)
    # print(solution_str)
    
    # 最终答案解析错误的直接判负
    try: eval(final_answer)
    except: final_answer = None if final_answer != 'no solution' else final_answer
    
    # 判断有多少数字能对上
    final_answer = '' if not final_answer else final_answer
    nums = sorted([int(n) for n in re.findall('\\d+', final_answer)])
    match_nums, error_nums = 0, 0
    for n in sorted([int(n) for n in extra_info['question'].split(',')]):
        while len(nums) and nums[0] < n:
            nums.pop(0)
            error_nums += 1
        if len(nums) and n == nums[0]:
            match_nums += 1
            nums.pop(0)
    error_nums += len(nums)
    final_answer = 0 if final_answer == 'no solution' else int(eval(final_answer)) if final_answer else -1
    return final_answer, match_nums - error_nums

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
    answer, match_nums = extract_solution(solution_str=solution_str, extra_info=extra_info, method=method)
    regex_format_ret = regext_format_score(solution_str)
    
    NUMS_LIMIT = 4
    assert NUMS_LIMIT < 24, "nums count over 24 point!"
    
    total_score = 0
    
    if answer == -1:
        total_score = 0
    else:
        if (answer == ground_truth or answer == 24) and match_nums == NUMS_LIMIT:
            if answer == 24 and ground_truth == 0 and match_nums == NUMS_LIMIT:
                print(solution_str)
            total_score = (0.7 if ground_truth == 0 else 1.0) * score
        else:
            total_score = match_nums / NUMS_LIMIT * format_score
    
    answer_score = total_score
    
    total_score += regex_format_ret * format_score
    
    # print(answer, match_nums, ground_truth, extra_info['question'], answer_score, total_score)
    
    answer_flexible, match_nums_flexible = extract_solution(solution_str=solution_str, extra_info=extra_info, method='flexible')
    correctness = score if (answer == ground_truth or answer == 24) and match_nums == NUMS_LIMIT else 0
    formatness = (match_nums / NUMS_LIMIT + regex_format_ret) * format_score
    
#     print(correctness, formatness, total_score)
#     print(solution_str)
    
    return total_score, correctness, formatness