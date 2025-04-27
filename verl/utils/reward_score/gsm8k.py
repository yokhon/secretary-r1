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


TAG_WORD = 'math_exp'

def extract_solution(solution_str, method='strict'):
    assert method in ['strict', 'flexible']

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    if len(matches) == 0:
        return None

    raw_answer = matches[-1].group(1).strip()

    if method == 'strict':
        # this also tests the formatting of the model
        # solution = re.search("(\\-?[0-9\\.\\,]+)", raw_answer)
        answer = re.search(r"(-?[\d.,]+)", raw_answer)
        if answer is None:
            final_answer = None
        else:
            final_answer = answer.group(0)
            final_answer = final_answer.replace(',', '').replace('$', '')
    elif method == 'flexible':
        # answer = re.findall("(\\-?[0-9\\.\\,]+)", raw_answer)
        answer = re.findall(r"(-?[\d.,]+)", raw_answer)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


def is_string_valid_for_calculator(input_str):
    pattern = r'^[x0-9+*/\-(). ]+$'
    return re.fullmatch(pattern, input_str) is not None


def correct_tag_format(solution_str, tag):
    pattern = r'<(%s)>(.*?)</\1>' % tag
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        content = match.group(2).strip()  # Return only the content inside the tags
        if not is_string_valid_for_calculator(content):
            # only works for calculator
            return False
        return True if len(content) > 0 and content != 'and' and content != '...' else False
    else:
        return False


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    # if answer is None:
    #     return 0
    # else:
    #     if answer == ground_truth:
    #         return score
    #     else:
    #         return format_score
    correct_query_format = correct_tag_format(solution_str, TAG_WORD)
    # correct_answer = correct_tag_format(solution_str, 'answer')
    correct_answer_format = True if answer is not None and len(answer) > 0 else False
    total_format_score = format_score * correct_query_format * 0.5 + format_score * correct_answer_format * 0.0
    answer_score = score if answer == ground_truth else 0.0
    total_score = answer_score + total_format_score
    return total_score
    # if answer == ground_truth:
    #     return score
    # else:
    #     return total_format_score