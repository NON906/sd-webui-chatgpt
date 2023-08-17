#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import openai
import sys
import json

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

    def load_log(self, log):
        if log is None:
            return False
        try:
            self.log_file_name = log
            if os.path.isfile(log):
                with open(log, 'r', encoding='UTF-8') as f:
                    self.chatgpt_messages = json.loads(f.read())
                return True
        except:
            pass
        return False

    def get_log(self):
        return json.dumps(self.chatgpt_messages)

    def write_log(self, file_name=None):
        if file_name is None:
            file_name = self.log_file_name
        if file_name is None:
            return
        with open(file_name + '.tmp', 'w', encoding='UTF-8') as f:
            f.write(self.get_log())
        if os.path.isfile(file_name):
            os.rename(file_name, file_name + '.prev')
        os.rename(file_name + '.tmp', file_name)
        if os.path.isfile(file_name + '.prev'):
            os.remove(file_name + '.prev')

    def send_to_chatgpt(self, content, write_log=False):
        if self.chatgpt_response is not None:
            return None
        self.chatgpt_messages.append({"role": "user", "content": content})
        self.chatgpt_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.chatgpt_messages,
            functions=self.chatgpt_functions
        )
        ignore_result = False
        result = str(self.chatgpt_response["choices"][0]["message"]["content"])
        prompt = None
        if "function_call" in self.chatgpt_response["choices"][0]["message"].keys():
            function_call = self.chatgpt_response["choices"][0]["message"]["function_call"]
            if function_call is not None and function_call["name"] == "txt2img":
                func_args = json.loads(function_call["arguments"])
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
        #print(result, file=sys.stderr)
        if write_log:
            self.write_log()
        if ignore_result:
            result = None
        return result, prompt

    def remove_last_conversation(self, result=None, write_log=False):
        if result is None or self.chatgpt_messages[-1]["content"] == result:
            self.chatgpt_messages = self.chatgpt_messages[:-2]
            if write_log:
                self.write_log()

    def clear(self):
        self.chatgpt_messages = []
        self.chatgpt_response = None
        self.log_file_name = None