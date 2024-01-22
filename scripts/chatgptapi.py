#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import openai
import sys
import json
from gpt_stream_parser import force_parse_json

class ChatGptApi:
    chatgpt_messages = []
    chatgpt_response = None
    log_file_name = None
    chatgpt_functions = [{
        "name": "txt2img",
        "description": "Generate image from prompt by Stable Diffusion. (Sentences cannot be generated.) There is no memory function, so please carry over the prompts from past conversations.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": 'Chat message. Displayed before the image.',
                },
                "prompt": {
                    "type": "string",
                    "description": 'Prompt for generate image. Prompt is comma separated keywords such as "1girl, school uniform, red ribbon". If it is not in English, please translate it into English (lang:en).',
                },
            },
            "required": ["prompt"],
        },
    }]
    model = 'gpt-3.5-turbo'
    recieved_json = ''
    recieved_message = ''

    def __init__(self, model=None, apikey=None):
        if model is not None:
            self.change_model(model)
        if apikey is not None:
            self.change_apikey(apikey)

    def change_apikey(self, apikey):
        openai.api_key = apikey

    def change_model(self, model):
        self.model = model

    def set_log(self, log_string):
        self.chatgpt_messages = json.loads(log_string)

    def get_log(self):
        return json.dumps(self.chatgpt_messages)

    def send(self, content):
        if self.chatgpt_response is not None:
            return None
        self.chatgpt_messages.append({"role": "user", "content": content})
        self.chatgpt_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.chatgpt_messages,
            functions=self.chatgpt_functions,
            stream=True,
        )
        ignore_result = False
        self.recieved_json = ''
        self.recieved_message = ''
        for chunk in self.chatgpt_response:
            if 'function_call' in chunk.choices[0].delta and chunk.choices[0].delta.function_call is not None and 'arguments' in chunk.choices[0].delta.function_call:
                self.recieved_json += chunk.choices[0].delta.function_call.arguments
            else:
                self.recieved_message += chunk.choices[0].delta.get('content', '')
        result = self.recieved_message
        prompt = None
        if self.recieved_json != '':
            func_args = json.loads(self.recieved_json)
            prompt = func_args["prompt"]
            if "message" in func_args:
                result = func_args["message"]
            else:
                ignore_result = True
        self.chatgpt_response = None
        if prompt is None:
            self.chatgpt_messages.append({"role": "assistant", "content": result})
        #elif ignore_result:
        #    self.chatgpt_messages.append({"role": "assistant", "content": "(Generated image by the following prompt: " + prompt + ")"})
        else:
            self.chatgpt_messages.append({"role": "assistant", "content": result + "\n(Generated image by the following prompt: " + prompt + ")"})
            result += '\n_' + prompt + '_'
        #print(result, file=sys.stderr)
        if ignore_result:
            result = '_' + prompt + '_'
            #result = None
        return result, prompt

    def remove_last_conversation(self, result=None):
        if result is None or self.chatgpt_messages[-1]["content"] == result:
            self.chatgpt_messages = self.chatgpt_messages[:-2]

    def clear(self):
        self.chatgpt_messages = []
        self.chatgpt_response = None
        self.log_file_name = None

    def get_stream(self):
        if self.recieved_json == '':
            return self.recieved_message, None
        func_args = force_parse_json(self.recieved_json)
        if func_args is not None and "message" in func_args:
            if "prompt" in func_args:
                return func_args["message"], func_args["prompt"]
            else:
                return func_args["message"], None
        else:
            return None, None
