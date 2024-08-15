import json
import argparse

def checkNum(s):
    for i in s:
      if i.isdigit():
         return True
    return False

def returnNum(s):
    l = []
    for i in s:
      if i.isdigit():
         l.append(i)
    return int("".join(l))


def check_eq(num1, num2):
    number=['1','2','3','4','5','6','7','8','9', '1/2']
    word=['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'half']
    if num1[0]=='$' or num1[0]=='(':
        num1=num1[1:]
    if num2[0]=='$':
        num2=num2[1:]
    if num1[-1]=='.' or num1[-1]==',':
        num1=num1[:-1]
    if num2[-1]=='.' or num2[-1]==',':
        num2=num2[:-1]
    if num1==num2:
       return 1
    
    else:
        for i in range(len(number)):
            if (num1==number[i] and num2==word[i]) or \
            (num1==word[i] and num2==number[i]):
                return 1

        return 0


parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="path to data file", type=str)
#parser.add_argument("model", help="model name", type=str)
#parser.add_argument("out_path", help="path to output file", type=str)
#parser.add_argument("negs", help="number of examples to generate", type=int)

args = parser.parse_args()

# with open(args.data_path, encoding='utf-8') as file:
#   data=[json.loads(line) for line in file.readlines()]
with open(args.data_path ) as file:
   data=json.load(file)
print(data['0'])


zero_shot_ans=[]

# out_dict = json.loads(open(output_path, 'r').read())
# # print(len(out_dict))
# i=len(out_dict)

# print(i, out_dict[0])    

acc=0

for i in range(len(data)):
  #print(i)
  question = data[str(i)]['original_question']
  f_question = data[str(i)]['final_question']
  # print(question)
  
  l = question.split(" ")
  b = False
  for j in range(len(l)):
     if(checkNum(l[j]) or l[j] in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'half', 'one.', 'two.', 'three.', 'four.', 'five.', 'six.', 'seven.', 'eight.', 'nine.', 'ten.', 'half.']):
        if not b:
            b=True
        else:
            ans=l[j]
            break
  l = f_question.split(" ")
  b = False
  for j in range(len(l)):
     if(checkNum(l[j]) or l[j] in ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'half', 'one.', 'two.', 'three.', 'four.', 'five.', 'six.', 'seven.', 'eight.', 'nine.', 'ten.', 'half.']):
        if not b:
            b=True
        else:    
            if(l[j]==ans):
               acc+=1
            else:
               val=check_eq(ans, l[j])
               acc+=val
               if val ==0:
                    print(i, ans, l[j])
               
            break

print(acc, acc/len(data))
