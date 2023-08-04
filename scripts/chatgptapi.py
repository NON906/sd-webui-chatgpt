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
        "description": "Generate image from prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": 'Prompts for generate images. Prompt is comma separated keywords such as "1girl, school uniform, red ribbon". Recommend around 30 keywords.',
                },
            },
            "required": ["prompt"],
        },
    }]

    def __init__(self, apikey=None):
        if apikey is not None:
            self.change_apikey(apikey)

    def change_apikey(self, apikey):
        openai.api_key = apikey

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

    def write_log(self):
        if self.log_file_name is None:
            return        
        with open(self.log_file_name + '.tmp', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(self.chatgpt_messages, sort_keys=True, indent=4, ensure_ascii=False))
        if os.path.isfile(self.log_file_name):
            os.rename(self.log_file_name, self.log_file_name + '.prev')
        os.rename(self.log_file_name + '.tmp', self.log_file_name)
        if os.path.isfile(self.log_file_name + '.prev'):
            os.remove(self.log_file_name + '.prev')

    def send_to_chatgpt(self, content, write_log=False):
        if self.chatgpt_response is not None:
            return None
        self.chatgpt_messages.append({"role": "user", "content": content})
        self.chatgpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.chatgpt_messages,
            functions=self.chatgpt_functions
        )
        result = str(self.chatgpt_response["choices"][0]["message"]["content"])
        prompt = None
        function_call = self.chatgpt_response["choices"][0]["message"]["function_call"]
        if function_call is not None and function_call["name"] == "txt2img":
            prompt = json.loads(function_call["arguments"])["prompt"]
        self.chatgpt_response = None
        self.chatgpt_messages.append({"role": "assistant", "content": result})
        #print(result, file=sys.stderr)
        if write_log:
            self.write_log()
        return result, prompt

    def remove_last_conversation(self, result=None, write_log=False):
        if result is None or self.chatgpt_messages[-1]["content"] == result:
            self.chatgpt_messages = self.chatgpt_messages[:-2]
            if write_log:
                self.write_log()

