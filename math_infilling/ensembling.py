import numpy as np
import pandas as pd
from scipy.special import binom, factorial
import json


gsm8k = [json.loads(a) for a in open('datasets/bot_output/gsm8k.json').readlines()]
multiarith = [json.loads(a) for a in open('datasets/bot_output/multiarith.json').readlines()]#[100:]
svamp = [json.loads(a) for a in open('datasets/bot_output/svamp.json').readlines()]#[100:]


tpr = 0.7594
fpr = 0.0739
fnr = 0.2405
tnr = 0.9261

#phrase_use = [0.99, 0.01, 0.01, 0.99]
#round_off = [0.7, 0.07, 0.3, 0.93]

dataset = gsm8k

        
holdout = gsm8k[:100]
#svamp = svamp[100:]
#multiarith = multiarith[100:]

def frequency_prior(eg):
    A = np.array(eg['answer_freq_mat'])
    prior = A.sum(axis=1)
    
    return prior / prior.sum()
    
def verify(eg, prior):
    # bayes rule
    u = len(eg['answer_idxs'])
    A = np.array(eg['verifier_results'])
    post = np.zeros(u)
    for i in range(u):
        if A[i] == 1:
            post[i] = (prior[i]*tpr)/(prior[i]*tpr + (1-prior[i])*fpr)
        else:
            post[i] = (prior[i]*fnr)/(prior[i]*fnr + (1-prior[i])*tnr)
    
    return post

correct = 0
count = 0
dataset = gsm8k[100:]
for eg in dataset:
    prior = frequency_prior(eg)
    inv_ans_idx = {b:a for a,b in eg['answer_idxs'].items()}
    if float(inv_ans_idx[np.argmax(verify(eg,prior))]) == float(eg['blanks'][0]): 
        correct += 1
    count += 1


print(correct/count)