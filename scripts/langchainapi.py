#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import Optional
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
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


class LangChainApi:
    log_file_name = None
    is_sending = False

    def __init__(self, **kwargs):
        self.backend = None
        self.memory = None

        self.load_settings(**kwargs)

    def init_model(self):
        if self.backend is None:
            return

        if self.backend == 'GPT4All':
            if (not 'gpt4all_model' in self.settings) or (self.settings['gpt4all_model'] is None):
                return
            self.llm = GPT4All(model=self.settings['gpt4all_model'])
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
                #verbose=True,
            )
            is_chat = False

        self.memory = ConversationBufferMemory(
            human_prefix="Human",
            ai_prefix="AI", 
            memory_key="history",
            return_messages=True,
        )

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
    "message": "This is a school girl wearing a red ribbon."
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

            human_template = "{human_input}<|end_of_turn|>AI:"
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template,
            )

            self.prompt = ChatPromptTemplate.from_messages([
                system_message_prompt,
                MessagesPlaceholder(variable_name="history"),
                human_message_prompt,
            ])

            self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm, memory=self.memory)#, verbose=True)

            def chat_predict(human_input):
                ret = self.llm_chain.invoke({
                    'human_input': human_input,
                })['text']
                #print(ret)
                return ret

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
                    history.add_ai_message(mes['content'])
            self.memory.chat_memory = history
        elif chatgpt_messages['log_version'] == 2:
            self.memory.chat_memory = messages_from_dict(chatgpt_messages['messages'])

    def load_log(self, log):
        if log is None:
            return False
        try:
            self.log_file_name = log
            if os.path.isfile(log):
                with open(log, 'r', encoding='UTF-8') as f:
                    self.set_log(f.read())
                return True
        except:
            pass
        return False

    def get_log(self):
        dicts = {'log_version': 2}
        if self.memory is None:
            dicts['messages'] = {}
        else:
            dicts['messages'] = messages_to_dict(self.memory.chat_memory)
        return json.dumps(dicts)

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

    def send(self, content, write_log=False):
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

        if write_log:
            self.write_log()

        self.is_sending = False

        return return_message, return_prompt

    def remove_last_conversation(self, result=None, write_log=False):
        if result is None or self.memory.chat_memory.messages[-1].content == result:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-2]
            if write_log:
                self.write_log()

    def clear(self):
        self.memory.chat_memory.clear()
        self.log_file_name = None