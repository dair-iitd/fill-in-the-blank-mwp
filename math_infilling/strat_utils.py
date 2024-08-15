import re
import logging
#from math_infilling.model import ChatCompletionModel
from collections import Counter
from sympy import solve, Eq
from sympy.abc import x, y, z   
import numpy as np
from sympy import sympify, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import traceback

import json


def extract_ans(completion):
    logger = logging.getLogger('rephrase')
    if completion is None:
        print("ERROR: Received no response")
        return None

    no_cot_group = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', completion.strip(), re.DOTALL)
    answer_group = re.match(r'.* ((-)?\d+(\.\d+)?)\.$', completion.strip(), re.DOTALL)
    fraction_group = re.match(r'.* -?(\d+)/(\d+)\.$', completion.strip(), re.DOTALL)
    no_cot_group_last = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', completion.strip().split(" ")[-1], re.DOTALL)

    if answer_group:
        answer = to_num(answer_group.group(1))
    elif no_cot_group:
        print("ISSUE : Didn't follow cot")
        #logger.warn("Didn't follow cot")
        answer = to_num(no_cot_group.group(1))
    elif fraction_group:
        answer = to_num(fraction_group.group(1))/to_num(fraction_group.group(2))
    elif no_cot_group_last:
        print("ISSUE Last: Didn't follow cot")
        #logger.warn("Didn't follow cot")
        answer = to_num(no_cot_group_last.group(1))
    else:
        print("ERROR: could not match returned question")
        logger.warn('Could not extract answer for response:')
        logger.warn(completion)
        return None
    logger.info(f"Obtained answer {completion}")
    return answer


def to_num(obj):
    if isinstance(obj, str):
        obj = obj.replace(',', '')
    try:
        return float(obj)
    except:
        return int(obj)



def reformat_incre_equations(result):
    x = re.findall(r'\[\[.*?\]\]', result)
    result = ''
    if len(x) >= 1:
        for eq in x:
            if len(result) == 0:
                result += eq[2 : -2]
            else:
                result += ', ' + eq[2 : -2]
    return result


def reformat_equations_from_peano(eq_list):
    result = ''
    for eq in eq_list.split(','):
        if 'eq' in eq:
            if len(result) == 0:
                result += eq[eq.index('eq') + 2:]
            else:
                result += ', ' + eq[eq.index('eq') + 2:]
        elif 'answer' in eq:
            if len(result) == 0:
                result += eq[eq.index('answer') + 6:].strip() + ' = ?'
            else:
                result += ', ' + eq[eq.index('answer') + 6:].strip() + ' = ?'     
    return result


def get_final_using_sympy(equations):
    logger = logging.getLogger('tools')
    try:
        transformations = (standard_transformations + (implicit_multiplication_application,) + (convert_xor,))
        if str(equations) == 'nan':
            logger.warn(f"ERROR in EQ {np.nan}")
            return None
        equation_list = equations.split(',')
        for eq in equation_list:
            for c in range(len(eq)):
                if c < len(eq) - 2:
                    if eq[c].isalpha() and eq[c+1].isalpha() and eq[c+2].isalpha():
                        logger.warn( 'ERROR in EQ  invalid equations p1')
                        return None

        goal_var = None
        goal_expression_list = []
            
        if equation_list[-1].split('=')[0].strip().isalpha() or len(equation_list[-1].split('=')[0].strip()) == 2:
            goal_var = equation_list[-1].split('=')[0].strip()
        elif '=' in equation_list[-1]:
            for l in list(string.ascii_lowercase) + list(string.ascii_uppercase):
                if l not in equation_list[-1]:
                    goal_var = l
                    break
            if goal_var is not None:
                goal_expression = goal_var + ' - (' + equation_list[-1].split('=')[0].strip() + ')'
                goal_expression = parse_expr(goal_expression, transformations=transformations)
                goal_expression = sympify(goal_expression)
                try:
                    return float(solve(goal_expression)[0])
                except Exception as e:
                    pass
                goal_expression_list.append(goal_expression)
            else:
                logger.warn( 'ERROR in EQ  invalid equations p2')
                return None

        if len(equation_list) == 1:
            try:
                goal_expression = parse_expr(equation_list[0].split('=')[0], transformations=transformations)
                return float(sympify(goal_expression))
            except Exception as e:
                logger.warn( 'ERROR in EQ  invalid equations p3')
                return None

        if goal_var == None:
            logger.warn( 'ERROR in EQ  no goal found')
            return None

        for i in range(len(equation_list) - 1):
            sub_eqs = equation_list[i]  
            if '?' not in sub_eqs:
                try:    
                    sub_eqs_split = sub_eqs.split('=')
                    sub_eqs = sub_eqs_split[0].strip() + ' - (' + sub_eqs_split[1].strip() + ')'
                    sub_eqs = parse_expr(sub_eqs, transformations=transformations)
                    sub_eqs = sympify(sub_eqs)
                except Exception as e:
                    logger.warn( 'ERROR in EQ  invalid equations p4')
                    return None
                goal_expression_list.append(sub_eqs)

                try:
                    try:
                        return float(solve(goal_expression_list)[Symbol(goal_var)])
                    except Exception as e:
                        return float(solve(goal_expression_list)[0][Symbol(goal_var)])
                except Exception as e:
                    pass

        logger.warn( 'ERROR in EQ  no sol')
        return None
    except Exception as e:
        logger.warn(e)
        logger.warn( 'ERROR in EQ  bug')
        return None
  



BLANK = "_____"

# def to_num(s):
#     try:
#         return int(s)
#     except:
#         return float(s)

def extract_blank(question, blank_limit = 1):
    tokens = re.split(r'\s+', question)
    first_num = True
    num_blanks_extracted = 0
    blanks = []

    numeric_tokens = {
            'one':   1,
            'two':   2,
            'three': 3,
            'four':  4,
            'five':  5, 
            'six':   6,
            'seven': 7,
            'eight': 8,
            'nine':  9,
            'zero':  0,
            'half':  0.5,
            'twice': 2,
            'thrice': 3
    }

    for idx, token in enumerate(tokens):
        if num_blanks_extracted >= blank_limit:
            break
        num_groups = re.match(r'[$]?(\d+(\.\d+)?)[.",%]?$', token)
        frac_groups = re.match(r'[$]?(\d+)/(\d+)[.",]?$', token)
        if num_groups:
            # numeric token
            if not first_num:
                tokens[idx] = re.sub(r'\d+(\.\d+)?', BLANK, token)
                num_blanks_extracted += 1
                blanks.append(to_num(num_groups.group(1)))

            first_num = False
        elif frac_groups:
            # fractional token
            if not first_num:
                tokens[idx] = re.sub(r'\d+/\d+', BLANK, token)
                num_blanks_extracted += 1
                blanks.append(to_num(frac_groups.group(1))/to_num(frac_groups.group(2)))

            first_num = False
        elif token in numeric_tokens:
            if not first_num:
                tokens[idx] = BLANK
                blanks.append(numeric_tokens[token])
                num_blanks_extracted += 1

            first_num = False

    if num_blanks_extracted < blank_limit:
        raise Exception(f'Could not extract {blank_limit} blanks, got only {num_blanks_extracted} blanks')

    return ' '.join(tokens), blanks







def extract_phrase_ans(completion):
    logger = logging.getLogger('rephrase')
    if completion is None:
        print("ERROR: Received no response")
        return None

    no_cot_group = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', completion.strip(), re.DOTALL)
    answer_group = re.match(r'.* ((-)?\d+(\.\d+)?)\.$', completion.strip(), re.DOTALL)
    fraction_group = re.match(r'.* -?(\d+)/(\d+)\.$', completion.strip(), re.DOTALL)
    no_cot_group_last = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', completion.strip().split(" ")[-1], re.DOTALL)

    if answer_group:
        answer = to_num(answer_group.group(1))
    elif no_cot_group:
        print("ISSUE : Didn't follow cot")
        #logger.warn("Didn't follow cot")
        answer = to_num(no_cot_group.group(1))
    elif fraction_group:
        answer = to_num(fraction_group.group(1))/to_num(fraction_group.group(2))
    elif no_cot_group_last:
        print("ISSUE Last: Didn't follow cot")
        #logger.warn("Didn't follow cot")
        answer = to_num(no_cot_group_last.group(1))
    else:
        tokens = re.split(r'\s+', completion)
        answer = None
        for t in reversed(tokens):
            t_no_cot_group = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', t.strip(), re.DOTALL)
            t_answer_group = re.match(r'.* ((-)?\d+(\.\d+)?)\.$', t.strip(), re.DOTALL)
            t_fraction_group = re.match(r'.* -?(\d+)/(\d+)\.$', t.strip(), re.DOTALL)
            t_no_cot_group_last = re.match(r'[$]?((-)?\d+(\.\d+)?)(\.)?$', t.strip().split(" ")[-1], re.DOTALL)
            
            if t_answer_group:
                answer = to_num(t_answer_group.group(1))
                break
            elif t_no_cot_group:
                print("ISSUE nl: Didn't follow cot")
                #logger.warn("Didn't follow cot")
                answer = to_num(t_no_cot_group.group(1))
                break
            elif t_fraction_group:
                answer = to_num(t_fraction_group.group(1))/to_num(t_fraction_group.group(2))
                break
            elif t_no_cot_group_last:
                print("ISSUE in loop: Didn't follow cot")
                #logger.warn("Didn't follow cot")
                answer = to_num(t_no_cot_group_last.group(1))
                break
            
        if answer is None:

            print("ERROR: could not match returned question")
            logger.warn('Could not extract answer for response:')
            logger.warn(completion)
            return None
    logger.info(f"Obtained answer {completion}")
    return answer

