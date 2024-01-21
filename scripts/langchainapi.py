#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
#from gpt_stream_parser import force_parse_json
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
    BaseMessage,
)
from langchain.chains import LLMChain, ConversationChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, List, Any
#from langchain.prompts.chat import (
#    ChatPromptTemplate,
#    SystemMessagePromptTemplate,
#    HumanMessagePromptTemplate,
#    AIMessagePromptTemplate,
#    MessagesPlaceholder,
#)
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
#from langchain_community.llms import OpenAI
#os.environ['OPENAI_API_KEY'] = 'foo'


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.recieved_message = ''

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.recieved_message += token


class TemplateMessagesPrompt(StringPromptTemplate):
    template: str = ''
    system_message: str = ''
    history_name: str = 'history'
    input_name: str = 'input'

    def format(self, **kwargs: Any) -> str:
        splited = self.template.split("{prompt}")
        human_template = splited[0] + "{prompt}"
        ai_template = splited[1] + "{response}\n"

        input_mes_list = kwargs[self.history_name]
        messages = self.system_message + '\n'
        for mes in input_mes_list:
            if type(mes) is HumanMessage:
                messages += human_template.replace("{prompt}", mes.content)
            elif type(mes) is AIMessage:
                messages += ai_template.replace("{response}", mes.content)
        messages += self.template.replace("{prompt}", kwargs[self.input_name])
        #print(messages)
        return messages


class LangChainApi:
    log_file_name = None
    is_sending = False

    def __init__(self, **kwargs):
        self.backend = None

        self.memory = ConversationBufferMemory(
            #human_prefix="Human",
            #ai_prefix="AI", 
            memory_key="history",
            return_messages=True,
        )
        self.callback = StreamingLLMCallbackHandler()

        self.load_settings(**kwargs)

    def init_model(self):
        if self.backend is None:
            return

        if self.backend == 'GPT4All':
            if (not 'gpt4all_model' in self.settings) or (self.settings['gpt4all_model'] is None):
                return
            self.llm = GPT4All(
                model=self.settings['gpt4all_model'],
                streaming=True,
                callback_manager=AsyncCallbackManager([self.callback]),
            )
            #self.llm = OpenAI(model_name="gpt-3.5-turbo")
            is_chat = False
            prompt_template_str = self.settings['gpt4all_prompt_template']
        if self.backend == 'LlamaCpp':
            if (not 'llama_cpp_model' in self.settings) or (self.settings['llama_cpp_model'] is None):
                return
            if not 'llama_cpp_n_gpu_layers' in self.settings:
                self.settings['llama_cpp_n_gpu_layers'] = 20
            if not 'llama_cpp_n_batch' in self.settings:
                self.settings['llama_cpp_n_batch'] = 128
            if not 'llama_cpp_n_ctx' in self.settings:
                self.settings['llama_cpp_n_ctx'] = 2048
            self.llm = LlamaCpp(
                model_path=self.settings['llama_cpp_model'],
                n_gpu_layers=self.settings['llama_cpp_n_gpu_layers'],
                n_batch=self.settings['llama_cpp_n_batch'],
                n_ctx=self.settings['llama_cpp_n_ctx'],
                streaming=True,
                callback_manager=AsyncCallbackManager([self.callback]),
                #verbose=True,
            )
            is_chat = False
            prompt_template_str = self.settings['llama_cpp_prompt_template']

        if not is_chat:
            system_message = """You are a chatbot having a conversation with a human.

You also have the function to generate image with Stable Diffusion.
If you want to use this function, please add the following to your message.

![PROMPT](sd-prompt:// "result")

PROMPT contains the prompt to generate the image.
Prompt is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.

![1girl, school uniform, red ribbon](sd-prompt:// "result")

This prompt is automatically deleted and is not visible to the human.
The image is always output at the end, not at the location where it is added.
If there are multiple entries, only the first one will be reflected.
There is no memory function, so please carry over the prompts from past conversations.
<|end_of_turn|>
If you understand, please reply to the following:<|end_of_turn|>
"""

            self.prompt = TemplateMessagesPrompt(
                system_message=system_message,
                template=prompt_template_str,
                input_variables=['history', 'input'],
            )

            self.llm_chain = ConversationChain(prompt=self.prompt, llm=self.llm, memory=self.memory)#, verbose=True)

            def chat_predict(human_input):
                ret = self.llm_chain.invoke({
                    'input': human_input,
                })
                #print(ret)
                return ret['response']

            self.chat_predict = chat_predict

        self.is_inited = True

    def load_settings(self, **kwargs):
        self.settings = kwargs
        self.backend = self.settings['backend']
        self.is_inited = False

    def set_log(self, log_string):
        chatgpt_messages = json.loads(log_string)
        if not 'log_version' in chatgpt_messages:
            history = ChatMessageHistory()
            for mes in chatgpt_messages:
                if mes['role'] == 'user':
                    history.add_user_message(mes['content'])
                elif mes['role'] == 'assistant':
                    if '\n(Generated image by the following prompt: ' in mes['content']:
                        mes_content, mes_prompt = mes['content'].split('\n(Generated image by the following prompt: ')
                        mes_prompt = mes_prompt[::-1].replace(')', '', 1)[::-1]
                        mes_json = json.dumps({
                            "prompt": mes_prompt,
                            "message": mes_content,
                        })
                    else:
                        mes_json = json.dumps({
                            "message": mes['content'],
                        })
                    history.add_ai_message(mes_json)
            self.memory.chat_memory = history
        elif chatgpt_messages['log_version'] == 2:
            self.memory.chat_memory = messages_from_dict(chatgpt_messages['messages'])


    def get_log(self):
        if self.memory is None:
            return '[]'
        ret_messages = []
        for mes in self.memory.chat_memory:
            if type(mes) is HumanMessage:
                ret_messages.append({"role": "user", "content": mes.content})
            elif type(mes) is AIMessage:
                mes_dict = json.loads(mes.content)
                add_mes = mes_dict['message']
                if 'prompt' in mes_dict and mes_dict['prompt'] is not None and mes_dict['prompt'] != "":
                    add_mes += "\n(Generated image by the following prompt: " + mes_dict['prompt'] + ")"
                ret_messages.append({"role": "assistant", "content": add_mes})
        return json.dumps(ret_messages)
        #dicts = {'log_version': 2}
        #if self.memory is None:
        #    dicts['messages'] = {}
        #else:
        #    dicts['messages'] = messages_to_dict(self.memory.chat_memory)
        #return json.dumps(dicts)

    def send(self, content):
        if not self.is_inited:
            self.init_model()

        if self.is_sending:
            return
        self.is_sending = True

        result = self.chat_predict(human_input=content)
        return_message, return_prompt = self.parse_message(result)

        self.is_sending = False
        self.callback.recieved_message = ''

        return return_message, return_prompt

    def remove_last_conversation(self, result=None):
        if result is None or self.memory.chat_memory.messages[-1].content == result:
            if len(self.memory.chat_memory.messages) > 2:
                self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-2]
            else:
                self.memory.chat_memory.messages.clear()

    def clear(self):
        self.memory.chat_memory.clear()
        self.log_file_name = None

    def get_stream(self):
        if self.callback is None or self.callback.recieved_message == '':
            return None, None
        return_message, return_prompt = self.parse_message(self.callback.recieved_message)
        if return_message is not None and len(return_message) > 0 and return_message[-1] == '!':
            return_message = return_message[:-1]
        if return_message is None or return_message.isspace():
            return_message = None
        return return_message, return_prompt

    def parse_message(self, full_message):
        if not '![' in full_message:
            return full_message, None
        prompt_tags = re.findall('\!\[.*?\]\(.*?\)', full_message)
        if len(prompt_tags) <= 0:
            end_index = full_message.rfind('![')
            return full_message[:end_index], None
        ret_message = full_message
        prompt = None
        for tag in prompt_tags:
            if '](sd-prompt:// "result")' in tag:
                if prompt is None:
                    prompt = tag[len('!['):-len('](sd-prompt:// "result")')]
                ret_message = ret_message.replace(tag, '')
        if ret_message.isspace():
            return None, prompt
        end_index = ret_message.rfind('![')
        if end_index >= 0:
            return ret_message[:end_index], prompt
        return ret_message, prompt