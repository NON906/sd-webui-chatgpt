#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from gpt_stream_parser import force_parse_json
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain.chains import LLMChain, ConversationChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
#from langchain_community.llms import OpenAI
#os.environ['OPENAI_API_KEY'] = 'foo'


class Txt2ImgModel(BaseModel):
    prompt: Optional[str] = Field(description='''Prompt for generate image.
Generate image from prompt by Stable Diffusion. (Sentences cannot be generated.)
There is no memory function, so please carry over the prompts from past conversations.
Prompt is comma separated keywords such as "1girl, school uniform, red ribbon" (not list).
If it is not in English, please translate it into English (lang:en).''')
    message: str = Field(description='''Chat message.
Please enter the content of your reply to me.
If prompt is exists, Displayed before the image.''')


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self):
        self.recieved_message = ''

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.recieved_message += token


class LangChainApi:
    log_file_name = None
    is_sending = False

    def __init__(self, **kwargs):
        self.backend = None

        self.memory = ConversationBufferMemory(
            human_prefix="Human",
            ai_prefix="AI", 
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
        if self.backend == 'LlamaCpp':
            if (not 'llama_cpp_model' in self.settings) or (self.settings['llama_cpp_model'] is None):
                return
            if not 'llama_cpp_n_gpu_layers' in self.settings:
                self.settings['llama_cpp_n_gpu_layers'] = 20
            if not 'llama_cpp_n_batch' in self.settings:
                self.settings['llama_cpp_n_batch'] = 128
            self.llm = LlamaCpp(
                model_path=self.settings['llama_cpp_model'],
                n_gpu_layers=self.settings['llama_cpp_n_gpu_layers'],
                n_batch=self.settings['llama_cpp_n_batch'],
                n_ctx=2048,
                streaming=True,
                callback_manager=AsyncCallbackManager([self.callback]),
                #verbose=True,
            )
            is_chat = False

        self.pydantic_parser = PydanticOutputParser(pydantic_object=Txt2ImgModel)

        if not is_chat:
            template = """
You are a chatbot having a conversation with a human.
You also have the function to generate image with Stable Diffusion.

{format_instructions}

Below is an example of the final output:
```
{{
    "prompt": "1girl, school uniform, red ribbon",
    "message": "This is a school girl wearing a red ribbon.\nWhat do you think of this image?"
}}
```
<|end_of_turn|>
If you understand, please reply to the following:<|end_of_turn|>
"""
            format_instructions = self.pydantic_parser.get_format_instructions()
            system_message_prompt = SystemMessagePromptTemplate.from_template(
                template,
                partial_variables={"format_instructions": format_instructions},
            )

            human_template = "{input}<|end_of_turn|>AI:"
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template,
            )

            self.prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                MessagesPlaceholder(variable_name="history"),
                human_message_prompt,
            ])

            self.llm_chain = ConversationChain(prompt=self.prompt, llm=self.llm, memory=self.memory)#, verbose=True)

            def chat_predict(human_input):
                ret = self.llm_chain.invoke({
                    'input': human_input,
                })
                #print(ret)
                return ret['response']

            self.chat_predict = chat_predict

        self.parser = OutputFixingParser.from_llm(
            parser=self.pydantic_parser,
            llm=self.llm,
        )

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
        try:
            parse_result = self.parser.parse(result)
            return_message = parse_result.message
            return_prompt = parse_result.prompt
        except:
            return_message = result
            return_prompt = None

        self.is_sending = False

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
        if self.callback is None:
            return None, None
        if '{' in self.callback.recieved_message:
            if '}' in self.callback.recieved_message:
                recieved_json = self.callback.recieved_message[self.callback.recieved_message.find('{'):self.callback.recieved_message.rfind('}') + 1]
            else:
                recieved_json = self.callback.recieved_message[self.callback.recieved_message.find('{'):]
            recieved_dict = force_parse_json(recieved_json)
            if recieved_dict is not None and "message" in recieved_dict:
                if "prompt" in recieved_dict:
                    return recieved_dict["message"], recieved_dict["prompt"]
                else:
                    return recieved_dict["message"], None
            else:
                return None, None
        else:
            return None, None