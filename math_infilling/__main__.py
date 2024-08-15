from math_infilling.logger import DataLogger
import math_infilling.models.openai as openai
import math_infilling.models.google as google
import math_infilling.models.llama as llama
from math_infilling.prompt import PromptGenerator
from math_infilling.dataset import JSONLDataset
import math_infilling.config as config
from tqdm.auto import tqdm
import logging
from datetime import datetime

from dotenv import load_dotenv
import os
import argparse
import time

def parse_args():

    parser = argparse.ArgumentParser(
            prog='math_infilling',
            description='Code for conducting experiments for the Constrained Abductive Reasoning Paper'
        )

    parser.add_argument('-m', '--models',   type=str, nargs="+", default=["gpt-3.5-turbo"])
    parser.add_argument('-p', '--prompts',  type=str, nargs="+", default=["rephrase_nm"])
    parser.add_argument('-d', '--datasets', type=str, nargs="+", default=["gsm8k_num_masked"])
    parser.add_argument('-l', '--llama-url', type=str, help="LLaMa API URL")
    parser.add_argument('-r', '--result-dir', type=str, default=f"results/run_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
    

    parser.add_argument('-s', '--split-start',   type=int, default=0)
    parser.add_argument('-e', '--split-end',     type=int, default=50)
    parser.add_argument('-i', '--interm',        type=int, default=10)
    
    return parser.parse_args()


def run(args,model_name, dataset_name, ds, pg, model, data_logger, logger ):
    for prompt_strategy in args.prompts:
        interm=args.interm

        data = {
            'correct': 0,
            'total': 0,
            'extracted': 0,
            'responses': []
        }
        logger.info(f"{model_name}:{dataset_name}:{prompt_strategy}")

        data_logger[model_name][dataset_name][prompt_strategy] = data
        for example in tqdm(ds):
            
            config.strategies[prompt_strategy](example, pg, model, data)
            interm-=1
            if interm==0:
                data_logger.save(args.result_dir)
                interm=args.interm

        logger.info(f"{data['total']} examples run")
        logger.info(f"Accuracy: {data['correct']/data['total']} ({data['correct']}/{data['total']})")




def run(args,model_name, dataset_name, ds, pg, model, data_logger, logger ):
    for prompt_strategy in args.prompts:
        interm=args.interm

        data = {
            'correct': 0,
            'total': 0,
            'extracted': 0,
            'responses': []
        }
        logger.info(f"{model_name}:{dataset_name}:{prompt_strategy}")

        data_logger[model_name][dataset_name][prompt_strategy] = data
        for example in tqdm(ds):
            
            config.strategies[prompt_strategy](example, pg, model, data)
            interm-=1
            if interm==0:
                data_logger.save(args.result_dir)
                interm=args.interm

        logger.info(f"{data['total']} examples run")
        logger.info(f"Accuracy: {data['correct']/data['total']} ({data['correct']}/{data['total']})")




def main():

    args = parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
    openai.setup_api_key(os.environ.get('OPENAI_API_KEY'))
    google.setup_api_key(os.environ.get('PALM_API_KEY'))
    llama.api_url = args.llama_url

    os.makedirs(args.result_dir)

    logging.basicConfig(
            filename=os.path.join(args.result_dir, 'logfile.log'),
            filemode='a',
            format='[%(asctime)s.%(msecs)d](%(name)s:%(levelname)s) %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )

    data_logger = DataLogger()
    pg = PromptGenerator('prompts')

    logger = logging.getLogger('main')
    
    

    for model_name in args.models:

        model_args = config.models[model_name].DEFAULT_ARGS.copy()
        model_args['model'] = model_name
        model = config.models[model_name](model_args)

        for dataset_name in args.datasets:

            ds = JSONLDataset(config.datasets[dataset_name])[args.split_start:args.split_end]
            run(args,model_name, dataset_name, ds, pg, model, data_logger, logger )

            

    
    data_logger.save(args.result_dir)

if __name__ == "__main__":
    main()
