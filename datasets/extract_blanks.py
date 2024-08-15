import json
import re
import argparse

BLANK = "_____"

def to_num(s):
    try:
        return int(s)
    except:
        return float(s)

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

def parse_args():

    parser = argparse.ArgumentParser(prog="extract_blanks")

    parser.add_argument('input', help='Input dataset filename')
    parser.add_argument('output', help='Output dataset filename')

    parser.add_argument('-k', '--key', help='name of key in JSON', default='question')
    parser.add_argument('-a', '--ans', help='name of answer in JSON', default='answer')
    parser.add_argument('-d', '--discard', help='discard fields already in dataset', action='store_true')
    parser.add_argument('-e', '--extracted', type=str, help='output examples that were extracted to a file', default='extracted.jsonl')

    return parser.parse_args()

def process_example(example, args):

    example = json.loads(example)
    processed = {}
    if not args.discard:
        processed = example.copy()

    blanked_q, blanks = extract_blank(example[args.key])

    processed[args.key] = blanked_q
    processed['blanks'] = blanks
    processed[args.ans] = example[args.ans].split(' ')[-1]

    return json.dumps(processed)

def extract_answer(example, args):

    example = json.loads(example)
    processed = {}
    if not args.discard:
        processed = example.copy()

    processed[args.key] = example[args.key]
    processed[args.ans] = example[args.ans].split(' ')[-1]

    return json.dumps(processed)

def main():

    args = parse_args()

    n_error = 0
    n_egs = 0

    unextracted_egs = []
    extracted_egs = []
    with open(args.input, 'r') as infile:
        with open(args.output, 'w') as outfile:
            for l in infile:
                try:
                    outfile.write(process_example(l, args) + '\n')
                    extracted_egs.append(l)
                except Exception as e:
                    n_error += 1
                    print(e)
                    print(f'Could not extract blanks for example {l}')
                    unextracted_egs.append(l)
                n_egs += 1

    with open(args.extracted, 'w') as extracted:
        for l in extracted_egs:
            extracted.write(extract_answer(l, args) + '\n')

    if n_error > 0:
        print(f'Could not extract blanks for {n_error}/{n_egs} examples')
    else:
        print(f'Extracted blanks for all {n_egs} examples')
                


if __name__ == "__main__":
    main()

