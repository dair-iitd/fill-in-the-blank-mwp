from math_infilling.strategies import *
from math_infilling.models.openai import GPT, ChatGPT
from math_infilling.models.google import PaLM
from math_infilling.models.llama import Llama2

strategies = {
    'cot_8shot': cot_8shot,
    'forward_cot_8shot': forward_cot_8shot,
    'rephrase_fewshot': rephrase_fewshot,
    'pal_tools': pal_tools,
    'pal' : pal,
    'rephrase_pal' : rephrase_pal,
    'rephrase_pal_tools' : rephrase_pal_tools,
    'tools' : tools,
    'rephrase_fewshot_sr' : rephrase_fewshot_sr,
    'check_your_work': check_your_work,
    'check_your_work_rephrase': check_your_work_rephrase,
    'self_refine_pal_tools': self_refine_pal_tools,
    'self_refine': self_refine,
    'forward_verify' : forward_verify,
    'bag_of_techniques': bag_of_techniques,
    'cot_8shot_phrase' : cot_8shot_phrase,
    'cot_8shot_phrase_verifyans' : cot_8shot_phrase_verifyans,
    'rephrase_fewshot_phrase' : rephrase_fewshot_phrase,
    'phrase_check_your_work' : phrase_check_your_work,
    'phrase_pal_tools' : phrase_pal_tools,
    'phrase_tools' : phrase_tools,
}

datasets = {
    'gsm8k_num_masked': 'datasets/gsm8k/gsm8k_test_num_masked.jsonl',
    'multiarith_num_masked': 'datasets/multiarith/multiarith_num_masked.jsonl',
    'svamp_num_masked': 'datasets/svamp/svamp_num_masked.jsonl',
    'svamp': 'datasets/svamp/svamp_preprocessed.jsonl',
    'multiarith': 'datasets/multiarith/multiarith_preprocessed.jsonl',
    'gsm8k': 'datasets/gsm8k/extracted.jsonl',
    'svamp_phrase_masked': 'datasets/svamp/svamp_phrase_masked.jsonl',
    'gsm8k_phrase_masked_shuffle_100' : 'datasets/gsm8k/gsm8k_test_phrase_masked_shuffle100.jsonl',
    'gsm8k_phrase_masked_shuffle2_100' : 'datasets/gsm8k/gsm8k_test_phrase_masked_shuffle2_100.jsonl',
    'svamp_phrase_masked_shuffle_100' : 'datasets/svamp/svamp_phrase_masked_shuffle_100.jsonl',
    'svamp_phrase_masked_shuffle2_100' : 'datasets/svamp/svamp_phrase_masked_shuffle2_100.jsonl',
    'multiarith_phrase_masked_shuffle_100' : 'datasets/multiarith/multiarith_phrase_masked_shuffle_100.jsonl',
    'multiarith_phrase_masked_shuffle2_100' : 'datasets/multiarith/multiarith_phrase_masked_shuffle2_100.jsonl',
}

models = {
    'text-davinci-003': GPT,
    'text-davinci-002': GPT,
    'gpt-35-turbo': ChatGPT,
    'gpt4': ChatGPT,
    'text-bison-001' : PaLM,
    'llama-2-70b' : Llama2
}
