#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from langchain import (
    hub,
)
from langchain_community.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    HumanMessage,
)
from langchain.agents import (
    Tool,
    create_react_agent,
    AgentExecutor,
)
#from langchain.llms import OpenAI
#os.environ['OPENAI_API_KEY'] = 'hoge'

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
            memory_key="chat_history",
            return_messages=True,
        )
        
        self.response = None

        def txt2img_func(prompt):
            self.response = prompt
            return "(Generated image by the following prompt: " + prompt + ")"

        self.tools = [
            Tool(
                name="txt2img",
                description="""
Generate image from prompt by Stable Diffusion. (Sentences cannot be generated.)
There is no memory function, so please carry over the prompts from past conversations.

Input prompt is comma separated keywords such as "1girl, school uniform, red ribbon".
If it is not in English, please translate it into English (lang:en).

Only the last input is valid.
Run it only once.
                """,
                func=txt2img_func,
            )
        ]

        if self.model_class == 'GPT4All':
            local_path = self.model
            self.llm = GPT4All(model=local_path, verbose=True)
            #self.llm = OpenAI(model_name="gpt-3.5-turbo", verbose=True)

            self.prompt = hub.pull("hwchase17/react")
            self.agent = create_react_agent(self.llm, self.tools, self.prompt)

            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                handle_parsing_errors=True,
                verbose=True,
            )

    def change_model_class(self, model_class):
        self.model_class = model_class
        self.init_model()

    def change_model(self, model):
        self.model = model
        self.init_model()

    def set_log(self, log_string):
        chatgpt_messages = json.loads(log_string)
        history = ChatMessageHistory()
        for mes in chatgpt_messages:
            if mes['role'] == 'user':
                history.add_user_message(mes['content'])
            elif mes['role'] == 'assistant':
                history.add_ai_message(mes['content'])
        self.memory.chat_memory = history

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
        chatgpt_messages = []
        for mes in self.memory.chat_memory.messages:
            if type(mes) is HumanMessage:
                chatgpt_messages.append({"role": "user", "content": mes.content})
            elif type(mes) is AIMessage:
                chatgpt_messages.append({"role": "assistant", "content": mes.content})
        return json.dumps(chatgpt_messages)

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

        self.response = None

        result = self.agent_executor.invoke({
            "input": content,
            "chat_history": self.memory.load_memory_variables({})['chat_history'],
        })['output']

        if self.response is not None:
            self.memory.chat_memory.messages[-1].content += "\n(Generated image by the following prompt: " + self.response + ")"

        if write_log:
            self.write_log()

        self.is_sending = False

        return result, self.response

    def remove_last_conversation(self, result=None, write_log=False):
        if result is None or self.memory.chat_memory.messages[-1].content == result:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[:-2]
            if write_log:
                self.write_log()

    def clear(self):
        self.memory.chat_memory.clear()
        self.log_file_name = None