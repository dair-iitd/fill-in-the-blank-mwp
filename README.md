# Fill in the blank Math Word Problem
This repo contains the dataset and code for the paper Fill in the Blank: Exploring and Enhancing LLM Capabilities for Backward Reasoning in Math Word Problems
The paper can be accessed at https://doi.org/10.48550/arXiv.2310.01991

## Environment Setup

```
conda create -n math_infill python=3.10
conda activate math_infill
pip install -e .
```

## Sample run
- To check Chain-of-Thought method's performance on backward reasoning task of GSM8k dataset, use the model as follows:
```
python -m math_infilling -p cot_8shot -d gsm8k_num_masked -s 0 -e 1272 -m gpt-35-turbo -r results/try_cot_gsm8k
```
- Please check the [config](https://github.com/dair-iitd/fill-in-the-blank-mwp/blob/main/math_infilling/config.py) file and the paper for more options. 