import openai
import logging
import tiktoken
import google.generativeai as palm
import time
import os
import backoff
from openai import AzureOpenAI

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds after {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))
    print(details)


@backoff.on_exception(backoff.expo, Exception, max_tries=10, on_backoff=backoff_hdlr)
@backoff.on_predicate(backoff.expo, lambda response: response is None)
def get_response(prompt, args, client=None, chat=False):
    if chat:
        response = client.chat.completions.create(**{'messages': prompt, **args})
    else:
        response = openai.Completion.create(**{'prompt': prompt, **args})

    return response

from math_infilling.model import TextCompletionModel, ChatCompletionModel

def setup_api_key(api_key):

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'

class GPT(TextCompletionModel):

    DEFAULT_ARGS = {
        'model': 'text-davinci-003',
        'max_tokens': 1024,
        'stop': None,
        'temperature': 0.5
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = GPT.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')
        self.enc = tiktoken.encoding_for_model(self.default_args['model'])

    def complete(self, prompt, args=None):

        response = None
        if not args:
            args = self.default_args

        try:
            response = get_response(prompt, args)
            
        except Exception as e:
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return None
            return response['choices'][0]['text'].strip()

    def get_num_tokens(self, prompt, args=None):

        return len(self.enc.encode(prompt))

class ChatGPT(ChatCompletionModel):

    DEFAULT_ARGS = {
        #'model': 'gpt-3.5-turbo',
        'model' : 'gpt-35-turbo',
        'max_tokens': 1024,
        'stop': None,
        'temperature': 0.5
    }

    def __init__(self, default_args=None):

        if default_args:
            self.default_args = default_args
        else:
            self.default_args = ChatGPT.DEFAULT_ARGS

        self.logger = logging.getLogger('GPT')

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.client = AzureOpenAI(
        api_version="2023-05-15",
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),

        )

    def complete(self, prompt, args=None):

        response = None

        if not args:
            args = self.default_args

        self.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        self.logger.info(f'Giving the following prompt:{prompt}')
        try:
            response = get_response(self.chat_history, args, self.client, chat=True)

        except Exception as e:
            
            print(e)
        finally:
            self.logger.info('Received the following response:')
            self.logger.info(response)
            if not response:
                return None
            #return response['choices'][0]['message']['content'].strip()
            return response.choices[0].message.content.strip()

    def clear(self):

        self.chat_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]


