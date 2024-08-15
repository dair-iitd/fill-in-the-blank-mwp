import json
import re
import argparse

BLANK = " _____"

def to_num(s):
    try:
        return int(s)
    except:
        return float(s)



      


def extract_sent(question, phrase_limit = 1):
    #tokens = re.split(r'\s+', question)
    first_num = True
    num_phrases_extracted = 0
    blanks = []
    masked_phrase = []

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

    phrase_list = []
    start = 0
    for j in range(len(question)):
        if(question[j] in ['.',','] and j+1<len(question) and question[j+1] == ' '):
            phrase_list.append(question[start:j])
            phrase_list.append(question[j])
            start = j+1
        elif(question[j:j+3] == 'and'):
            phrase_list.append(question[start:j-1])
            phrase_list.append(question[j-1:j+3])
            start = j+3
    if(start!=len(question)):
        phrase_list.append(question[start:])

    #print(phrase_list)
    #exit(0)
    for phid, phrase in enumerate(phrase_list):
        tokens = re.split(r'\s+', phrase)
        if num_phrases_extracted >= phrase_limit:
            break

        for idx, token in enumerate(tokens):
            if num_phrases_extracted >= phrase_limit:
                break
            num_groups = re.match(r'[$]?(\d+(\.\d+)?)[.",%]?$', token)
            frac_groups = re.match(r'[$]?(\d+)/(\d+)[.",]?$', token)
            if num_groups:
                # numeric token
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    num_phrases_extracted += 1
                    blanks.append(to_num(num_groups.group(1)))

                first_num = False
                break

            elif frac_groups:
                # fractional token
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    num_phrases_extracted += 1
                    blanks.append(to_num(frac_groups.group(1))/to_num(frac_groups.group(2)))

                first_num = False
                break
            elif token in numeric_tokens:
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    blanks.append(numeric_tokens[token])
                    num_phrases_extracted += 1

                first_num = False
                break

    if num_phrases_extracted < phrase_limit:
        #print("Issue")
        raise Exception(f'Could not extract {phrase_limit} phrases, got only {num_phrases_extracted} phrases')

    final_que = ""
    for phrase in phrase_list:
        final_que += phrase
        # if (phrase == ".") or (phrase == ",") or len(final_que)==0 :
        #     final_que += phrase
        # else:
        #     final_que += " "
        #     final_que += phrase

    return final_que, blanks, masked_phrase



def extract_phrase(question, phrase_limit = 1):
    #tokens = re.split(r'\s+', question)
    first_num = True
    num_phrases_extracted = 0
    blanks = []
    masked_phrase = []

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

    phrase_list = []
    start = 0
    for j in range(len(question)):
        if(question[j] in ['.',','] and j+1<len(question) and question[j+1] == ' '):
            phrase_list.append(question[start:j])
            phrase_list.append(question[j])
            start = j+1
        elif(question[j:j+3] == 'and'):
            phrase_list.append(question[start:j-1])
            phrase_list.append(question[j-1:j+3])
            start = j+3
    if(start!=len(question)):
        phrase_list.append(question[start:])

    #print(phrase_list)
    #exit(0)
    for phid, phrase in enumerate(phrase_list):
        tokens = re.split(r'\s+', phrase)
        if num_phrases_extracted >= phrase_limit:
            break

        for idx, token in enumerate(tokens):
            if num_phrases_extracted >= phrase_limit:
                break
            num_groups = re.match(r'[$]?(\d+(\.\d+)?)[.",%]?$', token)
            frac_groups = re.match(r'[$]?(\d+)/(\d+)[.",]?$', token)
            if num_groups:
                # numeric token
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    num_phrases_extracted += 1
                    blanks.append(to_num(num_groups.group(1)))

                first_num = False
                break

            elif frac_groups:
                # fractional token
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    num_phrases_extracted += 1
                    blanks.append(to_num(frac_groups.group(1))/to_num(frac_groups.group(2)))

                first_num = False
                break
            elif token in numeric_tokens:
                if not first_num:
                    masked_phrase.append(phrase_list[phid])
                    phrase_list[phid] = BLANK
                    blanks.append(numeric_tokens[token])
                    num_phrases_extracted += 1

                first_num = False
                break

    if num_phrases_extracted < phrase_limit:
        #print("Issue")
        raise Exception(f'Could not extract {phrase_limit} phrases, got only {num_phrases_extracted} phrases')

    final_que = ""
    for phrase in phrase_list:
        final_que += phrase
        # if (phrase == ".") or (phrase == ",") or len(final_que)==0 :
        #     final_que += phrase
        # else:
        #     final_que += " "
        #     final_que += phrase

    return final_que, blanks, masked_phrase


def parse_args():

    parser = argparse.ArgumentParser(prog="extract_phrases")

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

    blanked_q, blanks, masked_phrase = extract_phrase(example[args.key])

    processed[args.key] = blanked_q
    processed['blanks'] = blanks
    processed[args.ans] = example[args.ans].split(' ')[-1]
    processed['phrases'] = masked_phrase

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

