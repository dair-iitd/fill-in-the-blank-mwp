import json
from math_infilling.config import datasets

def get_acc_vector(out_file):
    
    for key, value in datasets.items():
        if out_file.find(key)!= -1:
            dataset_path = value
            print(f"Dataset: {value}")

    if dataset_path.find("num_masked")!= -1:
        masked=True

    examples = []
    with open(dataset_path, 'r') as file:
        for l in file:
            examples.append(json.loads(l))

    generations = []
    with open(out_file, 'r') as file:
        for l in file:
            generations.append(json.loads(l))


    j = 0
    acc_logs=[] # Reponsed? Extracted? Correct? 
    for i in range(len(generations)):
        question_gen = generations[i]['question']
        explanation = generations[i]['explanation']
        
        while (j<len(examples)):
            question = examples[j]['question']
            answer = examples[j]['answer']
            if masked:
                blank = examples[j]['blanks'][0]

            if question_gen == question:
                if 'extracted_answer' in generations[i]: 
                    extracted_ans = generations[i]['extracted_answer']
                    if masked and blank == extracted_ans:
                        acc_logs.append(1,1,1)
                    elif answer == extracted_ans:
                        acc_logs.append(1,1,1)
                    else:
                        acc_logs.append(1,1,0)
                
                else:
                    acc_logs.append(1,0,0)

            else:
                acc_logs.append(0,0,0)
            j+=1
        
        i+=1
    print(acc_logs)


