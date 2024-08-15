import sys
import json

def process_eg(example):

    example['question'] = example['sQuestion'].strip()
    example['answer'] = str(example['lSolutions'][0])

    del example['lSolutions']
    del example['sQuestion']

    return json.dumps(example)

def main():

    input = sys.argv[1]
    output = sys.argv[2]

    input_data = json.load(open(input, 'r'))
    # merge body and question together in svamp
    with open(output, 'w') as outfile:
        for example in input_data:
            outfile.write(process_eg(example) + '\n')

if __name__ == "__main__":
    main()
