#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import asyncio
#from gpt_stream_parser import force_parse_json
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
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

from langchain_community.chat_models.ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.recieved_message = ''
        self.is_cancel = False

    def on_chat_model_start(
        self, serialized, messages, **kwargs
    ):
        return

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.recieved_message += token
        if self.is_cancel:
            self.is_cancel = False
            raise asyncio.CancelledError


class TemplateMessagesPrompt(StringPromptTemplate):
    template: str = ''
    system_message: str = ''
    history_name: str = 'history'
    input_name: str = 'input'

    def format(self, **kwargs: Any) -> str:
        splited = self.template.split("{prompt}")
        human_template = splited[0]
        ai_template = splited[1]

        input_mes_list = kwargs[self.history_name]
        messages = self.system_message + '\n'
        for mes in input_mes_list:
            if type(mes) is HumanMessage:
                messages += human_template + mes.content
            elif type(mes) is AIMessage:
                messages += ai_template + mes.content + "\n"
        messages += self.template.replace("{prompt}", kwargs[self.input_name])
        #print(messages)
        return messages


class NewTemplateMessagesPrompt(StringPromptTemplate):
    full_template: str = ''
    human_template: str = ''
    ai_template: str = ''
    system_message: str = ''
    history_name: str = 'history'
    input_name: str = 'input'

    def format(self, **kwargs: Any) -> str:
        full_template = self.full_template.replace("{system}", self.system_message)
        human_template_before, human_template_after = self.human_template.split("{message}")
        ai_template_before, ai_template_after = self.ai_template.split("{message}")

        input_mes_list = kwargs[self.history_name]
        messages = ''
        for mes in input_mes_list:
            if type(mes) is HumanMessage:
                messages += human_template_before + mes.content + human_template_after
            elif type(mes) is AIMessage:
                messages += ai_template_before + mes.content + ai_template_after
        messages += human_template_before + kwargs[self.input_name] + human_template_after + ai_template_before
        full_messages = full_template.replace("{messages}", messages)
        #print(full_messages)
        return full_messages


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
            full_template_str = self.settings['llama_cpp_full_template']
            human_template_str = self.settings['llama_cpp_human_template']
            ai_template_str = self.settings['llama_cpp_ai_template']
            stop_word = ai_template_str.split("{message}")[-1]
            if not stop_word.isspace():
                stop_words = [stop_word, ]
            else:
                stop_words = []
            self.llm = LlamaCpp(
                model_path=self.settings['llama_cpp_model'],
                n_gpu_layers=int(self.settings['llama_cpp_n_gpu_layers']),
                n_batch=int(self.settings['llama_cpp_n_batch']),
                n_ctx=int(self.settings['llama_cpp_n_ctx']),
                streaming=True,
                callback_manager=AsyncCallbackManager([self.callback]),
                stop=stop_words,
                #verbose=True,
            )
            is_chat = False
        if self.backend == 'Ollama':
            if (not 'ollama_model' in self.settings) or (self.settings['ollama_model'] is None):
                return
            if not 'ollama_num_gpu' in self.settings:
                self.settings['ollama_num_gpu'] = 1
            if not 'llama_cpp_n_ctx' in self.settings:
                self.settings['llama_cpp_n_ctx'] = 2048
            if not 'ollama_keep_alive' in self.settings:
                self.settings['ollama_keep_alive'] = '5m'
            self.llm = ChatOllama(
                model=self.settings['ollama_model'],
                num_gpu=int(self.settings['ollama_num_gpu']),
                num_ctx=int(self.settings['llama_cpp_n_ctx']),
                keep_alive=self.settings['ollama_keep_alive'],
                streaming=True,
                callback_manager=AsyncCallbackManager([self.callback]),
                #verbose=True,
            )
            is_chat = True

        if 'llama_cpp_system_message_language' in self.settings and self.settings['llama_cpp_system_message_language'] == 'Japanese':
            system_message = """あなたは人間と会話するチャットボットです。

また、あなたはStable Diffusionで画像を生成する機能があります。
その機能を実行する場合は、以下をあなたの返信内容に加えてください。

![sd-prompt: PROMPT](sd:// "result")

PROMPTは画像生成に使用するプロンプトに置き換えてください。
このプロンプトはカンマ区切りの英語のキーワードの羅列です。
プロンプトが英語でない場合は、英語(lang:en)に翻訳してください。
例えば、「赤いリボンを付けた学生の女の子」を生成したい場合は、以下のようにしてください。

![sd-prompt: 1girl, school uniform, red ribbon](sd:// "result")

この画像は返信メッセージの後に表示されます。
このプロンプトが複数存在する場合は、最初のプロンプトの画像のみが生成されます。
この画像生成機能に、記憶する機能は無いので、過去の会話内容も反映させてください。
"""
        else:
            system_message = """You are a chatbot having a conversation with a human.

You also have the function to generate image with Stable Diffusion.
If you want to use this function, please add the following to your message.

![sd-prompt: PROMPT](sd:// "result")

PROMPT contains the prompt to generate the image.
Prompt is comma separated keywords.
If it is not in English, please translate it into English (lang:en).
For example, if you want to output "a school girl wearing a red ribbon", it would be as follows.

![sd-prompt: 1girl, school uniform, red ribbon](sd:// "result")

The image is always output at the end, not at the location where it is added.
If there are multiple entries, only the first one will be reflected.
There is no memory function, so please carry over the prompts from past conversations.
"""

        if not is_chat:
            if 'llama_cpp_system_message_language' in self.settings and self.settings['llama_cpp_system_message_language'] == 'Japanese':
                system_message += '<|end_of_turn|>\nこの内容を理解したら、以下の内容に返事をしてください:<|end_of_turn|>'
            else:
                system_message += '<|end_of_turn|>\nIf you understand, please reply to the following:<|end_of_turn|>'

            if self.backend == 'GPT4All':
                self.prompt = TemplateMessagesPrompt(
                    system_message=system_message,
                    template=prompt_template_str,
                    input_variables=['history', 'input'],
                )
            else:
                self.prompt = NewTemplateMessagesPrompt(
                    system_message=system_message,
                    full_template=full_template_str,
                    human_template=human_template_str,
                    ai_template=ai_template_str,
                    input_variables=['history', 'input'],
                )
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ])

        self.llm_chain = ConversationChain(prompt=self.prompt, llm=self.llm, memory=self.memory)#, verbose=True)

        def chat_predict(human_input):
            if self.callback.is_cancel:
                self.callback.is_cancel = False
                return None
            try:
                ret = self.llm_chain.invoke({
                    'input': human_input,
                })
            except asyncio.CancelledError:
                return None
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
        if type(chatgpt_messages) is not dict or not 'log_version' in chatgpt_messages:
            history = ChatMessageHistory()
            for mes in chatgpt_messages:
                if mes['role'] == 'user':
                    history.add_user_message(mes['content'])
                elif mes['role'] == 'assistant':
                    add_mes = re.sub("\(Generated image by the following prompt: (.*)\)", r'![sd-prompt: \1](sd:// "result")', mes['content'])
                    history.add_ai_message(add_mes)
            self.memory.chat_memory = history
        elif chatgpt_messages['log_version'] == 2:
            self.memory.chat_memory = messages_from_dict(chatgpt_messages['messages'])

    def get_log(self):
        if self.memory is None or self.memory.chat_memory is None:
            return '[]'
        ret_messages = []
        for name, messages in self.memory.chat_memory:
            if name == 'messages':
                for mes in messages:
                    if isinstance(mes, HumanMessage):
                        ret_messages.append({"role": "user", "content": mes.content})
                    elif isinstance(mes, AIMessage):
                        add_mes = re.sub('\!\[sd-prompt\: (.*?)\]\(sd\:// "result"\)', r'(Generated image by the following prompt: \1)', mes.content)
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
        if result is None:
            return_message, return_prompt = None, None
        else:
            return_message, return_prompt = self.parse_message(result, True)

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

    def parse_message(self, full_message, is_finished=False):
        if not '![' in full_message:
            return full_message, None
        prompt_tags = re.findall('\!\[.*?\]\(.*?\)', full_message)
        if len(prompt_tags) <= 0:
            if '![sd-prompt: ' in full_message:
                parted_prompt = full_message.split('![sd-prompt: ')[1]
                full_message = full_message.replace('![sd-prompt: ', '_').split(']')[0] + '_'
            end_index = full_message.rfind('![')
            if end_index >= 0:
                return full_message[:end_index], None
            if is_finished:
                return full_message, parted_prompt
            return full_message, None
        ret_message = full_message
        prompt = None
        for tag in prompt_tags:
            if tag.startswith('![sd-prompt: ') and tag.endswith('](sd:// "result")'):
                if prompt is None:
                    prompt = tag[len('![sd-prompt: '):-len('](sd:// "result")')]
                ret_message = ret_message.replace(tag, '_' + prompt + '_')
        if ret_message.isspace():
            return None, prompt
        if '![sd-prompt: ' in ret_message:
            ret_message = ret_message.replace('![sd-prompt: ', '_').split(']')[0] + '_'
        end_index = ret_message.rfind('![')
        if end_index >= 0:
            return ret_message[:end_index], prompt
        return ret_message, prompt

    def abort(self):
        self.callback.is_cancel = True