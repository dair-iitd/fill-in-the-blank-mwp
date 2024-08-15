import re
import logging
from math_infilling.model import ChatCompletionModel
from collections import Counter
from scipy.special import binom, logsumexp
from sympy import sympify, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import traceback

import traceback
from math_infilling.strat_utils import *
import random

def solution():
    pass


def pal(example, prompt_gen, model, data, dry_run=False):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'pal-A_chat')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return
    pycode = completion
    completion_match = re.match(r'(.*)Program:(.*)```python\n(.*)```', completion, re.DOTALL)
    program_match = re.match(r'(.*)```python\n(.*)```', completion, re.DOTALL)

    if completion_match is None and program_match is not None:
        rephrased_q = "No Response"
        pycode = program_match.group(2)
    elif completion_match is not None and program_match is not None:
        rephrased_q = completion_match.group(2)
        pycode = completion_match.group(3)

    globals_dict = {}
    locals_dict = {}

    pycode=pycode.strip()+"""
result = solution()"""
    logger.info(f"Executing: {pycode}")
    try:
        exec(pycode, globals_dict, locals_dict)
        extracted_answer = str(locals_dict.get('result', None))
    except Exception as e:
        logger.warn('Could not execute program and method for this question.')
        logger.warn(e)
        print("ERROR running program:")
        traceback.print_exc()

        if isinstance(model, ChatCompletionModel):
            model.clear()

        return

    data['total'] += 1
    if extracted_answer and (to_num(extracted_answer) == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'program': pycode,
        'extracted_answer': extracted_answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()





def rephrase_pal(example, prompt_gen, model, data, dry_run=False):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_pal-A6')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return
    pycode = completion
    completion_match = re.match(r'(.*)Program:(.*)```python\n(.*)```', completion, re.DOTALL)
    program_match = re.match(r'(.*)```python\n(.*)```', completion, re.DOTALL)

    if completion_match is None and program_match is not None:
        rephrased_q = "No Response"
        pycode = program_match.group(2)
    elif completion_match is not None and program_match is not None:
        rephrased_q = completion_match.group(2)
        pycode = completion_match.group(3)

    globals_dict = {}
    locals_dict = {}

    pycode=pycode.strip()+"""
result = finding_x()"""

    logger.info(f"Executing: {pycode}")
    try:
        exec(pycode, globals_dict, locals_dict)
        extracted_answer = str(locals_dict.get('result', None))
    except Exception as e:
        logger.warn('Could not execute program and method for this question.')
        logger.warn(e)
        print("ERROR running program:")
        traceback.print_exc()

        if isinstance(model, ChatCompletionModel):
            model.clear()

        return

    data['total'] += 1
    if extracted_answer and (to_num(extracted_answer) == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'rephrased_q' : rephrased_q,
        'program': pycode,
        'extracted_answer': extracted_answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()



def pal_tools(example, prompt_gen, model, data):

    logger = logging.getLogger('pal_tools')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'pal_tools_4shot')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return

    program_match = re.match(r'.*```python\n(.*)```.*', completion, re.DOTALL)

    if program_match is None:
        print("ERROR: could not extract program for question")
        logger.warn('Could not extract program for response:')
        logger.warn(completion)
        if isinstance(model, ChatCompletionModel):
            model.clear()
        return

    program = program_match.group(1)
    extracted_answer = -1

    # breakpoint()
    try:
        exec(program, globals())
        extracted_answer = to_num(solution())
    except Exception as e:
        logger.warn(f'Encountered exception {e} while executing the following program:')
        logger.warn(program)

        if isinstance(model, ChatCompletionModel):
            model.clear()

        return
    
    data['total'] += 1
    if (extracted_answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'program': program,
        'extracted_answer': extracted_answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()


def rephrase_pal_tools(example, prompt_gen, model, data):

    logger = logging.getLogger('pal_tools')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_pal_tools_4shot')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return

    completion_match = re.match(r'(.*)Program:(.*)```python\n(.*)```', completion, re.DOTALL)
    program_match = re.match(r'(.*)```python\n(.*)```', completion, re.DOTALL)

    if program_match is None:
        print("ERROR: could not extract program for question")
        logger.warn('Could not extract program for response:')
        logger.warn(completion)
        if isinstance(model, ChatCompletionModel):
            model.clear()
        return

    if completion_match is None and program_match is not None:
        rephrased_q = "No Response"
        program = program_match.group(2)
    elif completion_match is not None and program_match is not None:
        rephrased_q = completion_match.group(2)
        program = completion_match.group(3)


    # breakpoint()
    try:
        exec(program, globals())
        extracted_answer = to_num(solution())
    except Exception as e:
        logger.warn(f'Encountered exception {e} while executing the following program:')
        logger.warn(program)

        if isinstance(model, ChatCompletionModel):
            model.clear()

        return
    
    print(extracted_answer)

    data['total'] += 1
    if (extracted_answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'program': program,
        'rephrased_q' : rephrased_q,
        'extracted_answer': extracted_answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()




def tools(example, prompt_gen, model, data, rephrase=True):

    logger = logging.getLogger('tools')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    if rephrase:
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_tools')
    else:
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'tools_2')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return
    
    reformatted = reformat_incre_equations(completion.strip()+"\n\n\n")
    formatted_completion = reformat_equations_from_peano(reformatted)
    extracted_answer = get_final_using_sympy(formatted_completion)
    
    logger.info(completion)
    logger.info(f"Reformatted {reformatted}")
    logger.info(f"Formatted completion: {formatted_completion}")
    logger.info(f"extracted_answer : {extracted_answer}")

    data['total'] += 1
    if extracted_answer and (to_num(str(extracted_answer)) == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': formatted_completion,
        'extracted_answer': extracted_answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()



def rephrase_fewshot(example, prompt_gen, model, data, dry_run=False):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_fewshot')
    completion = model.complete(prompt)
    answer = extract_ans(completion)

    data['total'] += 1
    if (answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': completion,
        'extracted_answer': answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()


def rephrase_fewshot_sr(example, prompt_gen, model, data, fb_cnt=1): # Self Refine

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_fewshot')
    completion_init = model.complete(prompt)
    if isinstance(model, ChatCompletionModel):
        model.clear()

    while(fb_cnt>0):

        logger.info(f'Asking for feedback-{fb_cnt}')
        prompt = prompt_gen.create_sr_prompt(example['question'], example['answer'], 'fb_rephrase_fewshot', completion_init)
        completion_fb = model.complete(prompt)
        if isinstance(model, ChatCompletionModel):
            model.clear()

        prompt = prompt_gen.create_sr_prompt(example['question'], example['answer'], 'fb_up_rephrase_fewshot',
         completion_init, completion_fb, two=True) 
        completion = model.complete(prompt)
        completion_init=completion
        if isinstance(model, ChatCompletionModel):
            model.clear()

        fb_cnt-=1

    answer = extract_ans(completion_init)
    data['total'] += 1
    if (answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': completion,
        'extracted_answer': answer
    })


def cot_8shot(example, prompt_gen, model, data):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'cot_8shot_tc'
    # if isinstance(model, ChatCompletionModel):
    #     prompt_name = 'cot_8shot'

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], prompt_name)
    completion = model.complete(prompt)
    answer = extract_ans(completion)

    data['total'] += 1
    if (answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': completion,
        'extracted_answer': answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()

def forward_cot_8shot(example, prompt_gen, model, data):

    logger = logging.getLogger('cot')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'forward_cot_8shot_tc'
    # if isinstance(model, ChatCompletionModel):
    #     prompt_name = prompt_name[:-3]

    prompt = prompt_gen.create_q_prompt(example['question'], prompt_name)
    completion = model.complete(prompt)
    answer = extract_ans(completion)

    data['total'] += 1
    if (answer == to_num(example['answer'])):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': completion,
        'extracted_answer': answer
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()


def check_your_work(example, prompt_gen, model, data, max_iters=5, rephrase=False):

    logger = logging.getLogger('check_your_work')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'check_your_work'
    if rephrase:
        prompt_name = 'check_your_work_rephrase'

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], prompt_name)

    final_ans = -1
    ans_list = []
    comp_list = []

    iters = 0

    while iters < max_iters:
        completion = model.complete(prompt)
        comp_list.append(completion)

        response_match = re.match(r'(.*)Final question: (.*)Check: (.*)', completion, re.DOTALL)

        iters += 1

        if (response_match):
            answer = response_match.group(1)
            final_q = response_match.group(2)
            check = response_match.group(3)

            extracted_ans = extract_ans(answer)
            check_pass = 'This matches' in check

            ans_list.append(extracted_ans)
            if (check_pass):
                if extracted_ans:
                    final_ans = extracted_ans
                    if isinstance(model, ChatCompletionModel):
                        model.clear()
                    break

        if isinstance(model, ChatCompletionModel):
            model.clear()

    logger.info(f'ans_list: {ans_list}')

    data['total'] += 1
    if (final_ans >= 0):
        data['extracted'] += 1
        if (final_ans == to_num(example['blanks'][0])):
            data['correct'] += 1

    data['responses'].append({
        **example,
        'response': comp_list,
        'answer_list': ans_list,
        'extracted_answer': final_ans
    })

def check_your_work_rephrase(example, prompt_gen, model, data, max_iters=5):
    check_your_work(example, prompt_gen, model, data, max_iters=max_iters, rephrase=True)

def self_refine(example, prompt_gen, model, data, max_iters=5):

    logger = logging.getLogger('self_refine')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'self_refine_cot_init')

    final_ans = -1
    ans_list = []
    comp_list =[]

    cot = model.complete(prompt)
    comp_list.append(cot)
    if isinstance(model, ChatCompletionModel):
        model.clear()

    iters = 0

    while iters < max_iters:
        prompt = prompt_gen.create_prompt(
                'self_refine_cot_feedback', 
                question=example['question'], 
                answer=example['answer'], 
                chain_of_thought=cot
            )
        completion = model.complete(prompt)
        comp_list.append(completion)

        response_match = re.match(r'(.*)Final Solution:(.*)', completion, re.DOTALL)

        iters += 1

        if (response_match):
            corrections = response_match.group(1)
            cot = response_match.group(2)

            extracted_ans = extract_ans(cot)
            check_pass = ('completely correct' in corrections) and ('mistakes in the solution' not in corrections)

            ans_list.append(extracted_ans)
            if extracted_ans:
                final_ans = extracted_ans
                if (check_pass):
                    if isinstance(model, ChatCompletionModel):
                        model.clear()
                    break

        if isinstance(model, ChatCompletionModel):
            model.clear()

    logger.info(f'ans_list: {ans_list}')

    data['total'] += 1
    if (final_ans >= 0):
        data['extracted'] += 1
        if (final_ans == to_num(example['blanks'][0])):
            data['correct'] += 1

    data['responses'].append({
        **example,
        'response': comp_list,
        'answer_list': ans_list,
        'extracted_answer': final_ans
    })

def self_refine_pal_tools(example, prompt_gen, model, data, max_iters=5):

    logger = logging.getLogger('self_refine')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'self_refine_pal_tools_init')

    final_ans = -1
    ans_list = []
    comp_list =[]

    completion = model.complete(prompt)
    comp_list.append(completion)
    if isinstance(model, ChatCompletionModel):
        model.clear()
    
    response_match = re.match(r'(.*)Program:\n+```python(.*)```(.*)', completion, re.DOTALL)

    if not response_match:
        if isinstance(model, ChatCompletionModel):
            model.clear()
        print('ERROR: could not extract program. returning.')
        return

    rephrased = response_match.group(1)
    program = response_match.group(2)

    iters = 0

    while iters < max_iters:
        prompt = prompt_gen.create_prompt(
                'self_refine_pal_tools_feedback', 
                question=example['question'], 
                answer=example['answer'], 
                rephrased = rephrased,
                program = program
            )
        completion = model.complete(prompt)
        comp_list.append(completion)

        response_match = re.match(r'(.*)Final Rephrased Problem:\n(.*)\nFinal Program:\n+```python(.*)```(.*)', completion, re.DOTALL)

        iters += 1

        if (response_match):
            corrections = response_match.group(1)
            rephrased = response_match.group(2)
            program = response_match.group(3)

            check_pass = ('completely correct' in corrections)

            try:
                exec(program, globals())
                extracted_ans = to_num(solution())
            except Exception as e:
                logger.warn(f'Encountered exception {e} while executing the following program:')
                logger.warn(program)

                if isinstance(model, ChatCompletionModel):
                    model.clear()

                return

            ans_list.append(extracted_ans)
            if extracted_ans:
                final_ans = extracted_ans
                if (check_pass):
                    if isinstance(model, ChatCompletionModel):
                        model.clear()
                    break

        if isinstance(model, ChatCompletionModel):
            model.clear()

    logger.info(f'ans_list: {ans_list}')

    data['total'] += 1
    if (final_ans >= 0):
        data['extracted'] += 1
        if (final_ans == to_num(example['blanks'][0])):
            data['correct'] += 1

    data['responses'].append({
        **example,
        'response': comp_list,
        'answer_list': ans_list,
        'extracted_answer': final_ans
    })


def forward_verify(example, prompt_gen, model, data, give_correct):
    logger = logging.getLogger('Forward verify')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'forward_verify'
    # if isinstance(model, ChatCompletionModel):
    #     prompt_name = prompt_name[:-3]
    if give_correct:
        new_question = example['question'].replace('_____',
                                                   str(example['blanks'][0]))
    else:
        new_question = example['question'].replace('_____', 
                                                   str(example['blanks'][0]*(1+random.randint(1,10))))
    prompt = prompt_gen.create_qa_prompt(new_question, example['answer'], prompt_name)
    completion = model.complete(prompt)
    check_pass = 'This matches' in completion
    

    data['total'] += 1
    if (check_pass == give_correct) and give_correct:
        data['correct'] += 1
    elif (check_pass == give_correct) and not give_correct:
        data['F_correct'] += 1

    data['responses'].append({
        **example,
        'response': completion,
        'new_question' : new_question,
        'give_correct' : give_correct,
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()

def bag_of_techniques(example, prompt_gen, model, data, ADD1 = True):

    logger = logging.getLogger('bag_of_techniques')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    def rephrase_fewshot_technique():
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_fewshot')
        completion = model.complete(prompt)
        answer = extract_ans(completion)
        return answer

    def tools_technique():
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_tools')
        completion = model.complete(prompt)
        if completion is None:
            print("ERROR: could not extract ans for question")
            logger.warn('Could not extract program for response:')
            logger.warn(completion)
            return -10e9
        reformatted = reformat_incre_equations(completion.strip()+"\n\n\n")
        formatted_completion = reformat_equations_from_peano(reformatted)
        extracted_answer = get_final_using_sympy(formatted_completion)
        return extracted_answer

    def pal_tools_technique():
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'rephrase_pal_tools_4shot')
        completion = model.complete(prompt)
        program_match = re.match(r'.*```python\n(.*)```.*', completion, re.DOTALL)
        if program_match is None:
            print("ERROR: could not extract program for question")
            logger.warn('Could not extract program for response:')
            logger.warn(completion)
            return -10e9

        program = program_match.group(1)
        extracted_ans = -10e9

        try:
            exec(program, globals())
            extracted_ans = to_num(solution())
        except Exception as e:
            logger.warn(f'Encountered exception {e} while executing the following program:')
            logger.warn(program)
            return -10e9

        return extracted_ans

    techniques = {
        'rephrase_fewshot': rephrase_fewshot_technique, 
        'rephrase_tools': tools_technique, 
        'rephrase_pal_tools': pal_tools_technique
    }
    technique_idxs = {
        'rephrase_fewshot': 0, 
        'rephrase_tools': 1, 
        'rephrase_pal_tools': 2
    }

    sampling_paths_per_technique = 3

    blanks = { a : [] for a in techniques.keys() }

    for tname, technique in techniques.items():
        for _ in range(sampling_paths_per_technique):
            ans_returned = technique()
            if ans_returned:
                blanks[tname].append(ans_returned)
            else:
                print(f'Could not get answer from technique {tname}')
            if isinstance(model, ChatCompletionModel):
                model.clear()

    technique_accs = np.array([0.44, 0.37, 0.51])
    log_technique_accs = np.log(technique_accs)
    log_one_minus_technique_accs = np.log(1-technique_accs)
    log_verifier_tpr = np.log(0.7);
    log_verifier_fpr = np.log(0.07);
    log_verifier_fnr = np.log(0.03);
    log_verifier_tnr = np.log(0.93);

    answer_idxs = { a : j for j, a in enumerate(set([i for b in blanks.values() for i in b])) }
    inv_answer_idxs = { i : a for a, i in answer_idxs.items() }
    ans_freq_mat = np.zeros((len(answer_idxs), len(techniques)))

    for tname, answers in blanks.items():
        for answer in answers:
            ans_freq_mat[answer_idxs[answer]][technique_idxs[tname]] += 1

    # compute priors
    # Use a prior weighted by frequency

    log_ans_priors = np.zeros(ans_freq_mat.shape[0])
    for i, answer in enumerate(ans_freq_mat):
        for j, freq in enumerate(answer):
            log_ans_priors[i] += np.log(binom(sampling_paths_per_technique, freq)) + \
                                 freq * log_technique_accs[j] + \
                                 (sampling_paths_per_technique - freq) * log_one_minus_technique_accs[j]

    freq_scores = ans_freq_mat.sum(axis=1) / ans_freq_mat.sum()
    log_ans_priors += np.log(freq_scores)
    try:
        log_ans_priors -= logsumexp(log_ans_priors) # normalizing
    except: 
        print("error in normalizing")
        return

    print(f'Prior sum: {np.sum(np.exp(log_ans_priors))}')

    log_ans_inv_priors = np.log1p(-np.exp(log_ans_priors))

    log_ans_posteriors = np.zeros(log_ans_priors.shape)

    verifier_results = np.zeros(log_ans_priors.shape)

    for answer, i in answer_idxs.items():
        blank_repl = str(answer)
        if answer % 1 > 1e-6 and (1 - (answer % 1)) > 1e-6:
            blank_repl = f'{answer:.2}'
        new_question = example['question'].replace('_____',blank_repl)
        prompt = prompt_gen.create_qa_prompt(new_question, example['answer'], 'forward_verify')
        completion = model.complete(prompt)

        if not completion:
            print("ERROR: could not get completion for blank {blank}")
            if isinstance(model, ChatCompletionModel):
                model.clear()
            continue
        
        if 'this matches' in completion.lower():
            # verifier says it's correct 
            right_prob = log_ans_priors[i] + log_verifier_tpr
            wrong_prob = log_ans_inv_priors[i] + log_verifier_fpr
            log_ans_posteriors[i] = right_prob - np.logaddexp(right_prob, wrong_prob)
            verifier_results[i] = 1
        else:
            # assume wrong
            right_prob = log_ans_priors[i] + log_verifier_fnr
            wrong_prob = log_ans_inv_priors[i] + log_verifier_tnr
            log_ans_posteriors[i] = right_prob - np.logaddexp(right_prob, wrong_prob)

        if isinstance(model, ChatCompletionModel):
            model.clear()

    # print(log_ans_posteriors)
    # print(inv_answer_idxs)
    print(f'Posterior sum: {np.sum(np.exp(log_ans_posteriors))}')
        
    final_answer = inv_answer_idxs[np.argmax(log_ans_posteriors)]

    data['total'] += 1
    if (final_answer == example['blanks'][0]) and ADD1:
        data['correct'] += 1

    data['responses'].append({
        **example,
        'answer_idxs' : answer_idxs,
        'answer_freq_mat' : ans_freq_mat.tolist(),
        'log_ans_priors' : log_ans_priors.tolist(),
        'log_ans_posteriors' : log_ans_posteriors.tolist(),
        'final_ans': final_answer,
        'verifier_results': verifier_results.tolist()
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()

    return final_answer






def cot_8shot_phrase(example, prompt_gen, model, data, rephrase=False):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'cot_8shot_phrase'
    if rephrase:
        prompt_name = 'rephrase_fewshot_phrase_3'
    # if isinstance(model, ChatCompletionModel):
    #     prompt_name = 'cot_8shot'
    

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], prompt_name)
    completion = model.complete(prompt)
    phrase_list = completion.split(':')
    if len(phrase_list) > 1:
        phrase = phrase_list[1]
        if rephrase:
            phrase = phrase_list[2]
    else: 
        phrase = completion
    
    if len(phrase) > 1:
        answer = extract_phrase_ans(completion)

        data['total'] += 1
        if (to_num(example['blanks'][0]) == answer):
            data['correct'] += 1

        data['responses'].append({
            **example,
            'response': completion,
            'extracted_answer': answer,
            'phrase' : phrase
        })

    if isinstance(model, ChatCompletionModel):
        model.clear()



def cot_8shot_phrase_verifyans(example, prompt_gen, model, data):

    logger = logging.getLogger('rephrase')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'cot_8shot_phrase'
    # if isinstance(model, ChatCompletionModel):
    #     prompt_name = 'cot_8shot'
    

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], prompt_name)
    completion = model.complete(prompt)
    phrase_list = completion.split(':')
    if len(phrase_list) > 1:
        phrase = phrase_list[1]

        v_prompt_name = 'forward_verify'
        new_question = example['question'].replace('_____', phrase)
        v_prompt = prompt_gen.create_qa_prompt(new_question, example['answer'], v_prompt_name)
        v_completion = model.complete(v_prompt)
        check_pass = 'This matches' in v_completion
    
    # else: 
    #     phrase = completion
        
    
    # if len(phrase) > 1:
        answer = extract_phrase_ans(completion)

        data['total'] += 1
        if check_pass:
            data['correct'] += 1

        if (to_num(example['blanks'][0]) == answer):
            em = True
        else:
            em = False

        data['responses'].append({
            **example,
            'response': completion,
            'extracted_answer': answer,
            'phrase' : phrase,
            'extracted_match' : em,
            'v_completion' : v_completion,
        })

    if isinstance(model, ChatCompletionModel):
        model.clear()



 
def rephrase_fewshot_phrase(example, prompt_gen, model, data):

    cot_8shot_phrase(example, prompt_gen, model, data, rephrase=True)



def phrase_check_your_work(example, prompt_gen, model, data, max_iters=5, rephrase=False):


    logger = logging.getLogger('check_your_work')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    prompt_name = 'phrase_check_your_work_g'
    if rephrase:
        prompt_name = 'phrase_check_your_work_rephrase'

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], prompt_name)

    final_ans = -1
    ans_list = []
    comp_list = []

    iters = 0

    while iters < max_iters:
        completion = model.complete(prompt)
        comp_list.append(completion)

        response_match = re.match(r'(.*)Answer: (.*)Final question: (.*)Check: (.*)', completion, re.DOTALL)

        iters += 1

        if (response_match):
            guess = response_match.group(1)
            answer = response_match.group(2)
            final_q = response_match.group(3)
            check = response_match.group(4)

            extracted_ans = extract_ans(answer)
            check_pass = 'This matches' in check

            ans_list.append(extracted_ans)
            if (check_pass):
                if extracted_ans:
                    final_ans = extracted_ans
                    if isinstance(model, ChatCompletionModel):
                        model.clear()
                    break

        if isinstance(model, ChatCompletionModel):
            model.clear()

    logger.info(f'ans_list: {ans_list}')

    data['total'] += 1
    if (final_ans >= 0):
        data['extracted'] += 1
        if (final_ans == to_num(example['blanks'][0])):
            data['correct'] += 1

    data['responses'].append({
        **example,
        'response': comp_list,
        'answer_list': ans_list,
        'extracted_answer': final_ans
    })


def phrase_pal_tools(example, prompt_gen, model, data, rephrase= False):

    logger = logging.getLogger('pal_tools')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')
    
    pname = 'phrase_pal_tools'
    if rephrase:
        pname = 'phrase_rephrase_pal_tools'

    prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], pname) # 4 shot
    
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return

    program_match = re.match(r'.*```python\n(.*)```.*', completion, re.DOTALL)

    if program_match is None:
        print("ERROR: could not extract program for question")
        logger.warn('Could not extract program for response:')
        logger.warn(completion)
        if isinstance(model, ChatCompletionModel):
            model.clear()
        return

    program = program_match.group(1)
    extracted_answer = -1

    # breakpoint()
    try:
        exec(program, globals())
        extracted_answer = to_num(solution())
    except Exception as e:
        logger.warn(f'Encountered exception {e} while executing the following program:')
        logger.warn(program)

        if isinstance(model, ChatCompletionModel):
            model.clear()

        return
    
    data['total'] += 1
    if (extracted_answer == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'program': program,
        'extracted_answer': extracted_answer,
        'completion' : completion
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()



def phrase_tools(example, prompt_gen, model, data, rephrase=True):

    logger = logging.getLogger('tools')
    logger.info(f'Prompting with question and answer: Q) {example["question"]} A) {example["answer"]}')

    if rephrase:
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'phrase_rephrase_tools') #3s
    else:
        prompt = prompt_gen.create_qa_prompt(example['question'], example['answer'], 'phrase_rephrase_tools')
    completion = model.complete(prompt)

    if completion is None:
        print("ERROR: Received no response")
        return
    
    reformatted = reformat_incre_equations(completion.strip()+"\n\n\n")
    formatted_completion = reformat_equations_from_peano(reformatted)
    extracted_answer = get_final_using_sympy(formatted_completion)
    
    logger.info(completion)
    logger.info(f"Reformatted {reformatted}")
    logger.info(f"Formatted completion: {formatted_completion}")
    logger.info(f"extracted_answer : {extracted_answer}")

    data['total'] += 1
    if extracted_answer and (to_num(str(extracted_answer)) == example['blanks'][0]):
        data['correct'] += 1

    data['responses'].append({
        **example,
        'response': formatted_completion,
        'extracted_answer': extracted_answer,
        'completion' : completion
    })

    if isinstance(model, ChatCompletionModel):
        model.clear()



