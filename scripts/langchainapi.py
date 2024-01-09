#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from langchain import PromptTemplate
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from huggingface_hub import snapshot_download
#from langchain_community.llms import OpenAI
#os.environ['OPENAI_API_KEY'] = 'foo'

class Txt2ImgModel(BaseModel):
    message: str = Field(description='''Chat message.
If prompt is exists, Displayed before the image.''')
    prompt: Optional[str] = Field(description='''Prompt for generate image.
Generate image from prompt by Stable Diffusion. (Sentences cannot be generated.)
There is no memory function, so please carry over the prompts from past conversations.
Prompt is comma separated keywords such as "1girl, school uniform, red ribbon".
If it is not in English, please translate it into English (lang:en).'''
    )

class LangChainApi:
    log_file_name = None
    is_sending = False

    def __init__(self, model_class=None, model=None):
        self.model_class = None
        self.model = None

        if model_class is not None:
            self.change_model_class(model_class)
        if model is not None:
            self.change_model(model)

    def init_model(self):
        if self.model_class is None or self.model is None:
            return

        self.memory = ConversationBufferMemory(
            human_prefix="Human",
            ai_prefix="AI", 
            memory_key="chat_history",
            return_messages=True,
        )

        if self.model_class == 'GPT4All':
            local_path = self.model
            self.llm = GPT4All(model=local_path)
            #self.llm = OpenAI(model_name="gpt-3.5-turbo")
            is_chat = False
        if self.model_class == 'LlamaCpp':
            if os.path.isfile(self.model):
                local_path = self.model
            else:
                local_model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', self.model.replace('/', '_'))
                os.makedirs(local_model_dir, exist_ok=True)
                if len(os.listdir(local_model_dir)) == 0:
                    local_path = snapshot_download(
                        repo_id=self.model,
                        local_dir=local_model_dir,
                        local_dir_use_symlinks=False, 
                    )
                else:
                    local_path = local_model_dir
            self.llm = LlamaCpp(
                model_path=local_path,
                n_gpu_layers=20,
                n_batch=128,
                n_ctx=2048,
                verbose=True,
            )
            is_chat = False

        self.pydantic_parser = PydanticOutputParser(pydantic_object=Txt2ImgModel)

        if not is_chat:
            template = """You are a chatbot having a conversation with a human.
Answer the followimg format.
{format_instructions}
            
{chat_history}
Human: {human_input}
AI: """
            self.prompt = PromptTemplate(
                template=template,
                input_variables=["chat_history", "human_input"],
                partial_variables={"format_instructions": self.pydantic_parser.get_format_instructions()},
            )

            self.llm_chain = LLMChain(prompt=self.prompt, llm=self.llm, memory=self.memory, verbose=True)

            def chat_predict(human_input):
                return self.llm_chain.invoke({
                    'human_input': human_input,
                    #'chat_history': self.memory.load_memory_variables({})['chat_history'],
                })['text']

            self.chat_predict = chat_predict

        self.parser = RetryWithErrorOutputParser.from_llm(
            parser=self.pydantic_parser,
            llm=self.llm,
        )

    def change_model_class(self, model_class):
        self.model_class = model_class
        self.init_model()

    def change_model(self, model):
        self.model = model
        self.init_model()

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
        if self.is_sending:
            return
        self.is_sending = True

        result = self.chat_predict(human_input=content)
        parse_result = self.parser.parse_with_prompt(result, self.prompt.format_prompt(
            chat_history=self.memory.load_memory_variables({})['chat_history'],
            human_input=content,
        ))

        if parse_result.message.startswith('AI: '):
            parse_result.message = parse_result.message[4:]

        if write_log:
            self.write_log()

        self.is_sending = False

        return parse_result.message, parse_result.prompt

    def remove_last_conversation(self, result=None, write_log=False):
        if result is None or self.memory.chat_memory.messages[-1].content == result:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-2]
            if write_log:
                self.write_log()

    def clear(self):
        self.memory.chat_memory.clear()
        self.log_file_name = None