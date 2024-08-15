import sys
import json

def process_eg(example):

    if example['Body'][-1] == '.':
        example['question'] = example['Body'] + ' ' + example['Question']
    else:
        example['question'] = example['Body'] + ', ' + example['Question']

    example['answer'] = str(example['Answer'])

    del example['Body']
    del example['Question']
    del example['Answer']

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
