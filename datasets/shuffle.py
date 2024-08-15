import json
import re
import argparse
import random
import sys





def main():

    args = sys.argv
    
    check_data = []
    
    with open(args[4], 'r') as chfile:
        for l in chfile:
            check_data.append(l)
    print(len(check_data))
    
    in_data = []
    with open(args[1], 'r') as infile:
        for l in infile:
            in_data.append(l)
        random.shuffle(in_data)
        if int(args[3]) > 0:
            flen = int(args[3])
        else:
            flen = len(in_data)
        with open(args[2], 'w') as outfile:
            i = 0
            ocnt = 0 
            while ocnt < flen:
                if in_data[i] not in check_data:
                    outfile.write(in_data[i])
                    ocnt += 1
                    #print(i)
                i += 1
            
                

if __name__ == "__main__":
    main()