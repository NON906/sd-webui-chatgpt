#!/usr/bin/env python
# -*- coding: utf-8 -*-

import launch

if not launch.is_installed('openai'):
    launch.run_pip('install openai==0.28.1', 'openai')

if not launch.is_installed('langchain'):
    launch.run_pip('install langchain', 'langchain')

if not launch.is_installed('langchainhub'):
    launch.run_pip('install langchainhub', 'langchainhub')

if not launch.is_installed('gpt4all'):
    launch.run_pip('install gpt4all', 'gpt4all')

if not launch.is_installed('llama-cpp-python'):
    import os
    if os.name == 'nt':
        launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.23+cu118-cp310-cp310-win_amd64.whl', 'llama-cpp-python')
    else:
        launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.23+cu118-cp310-cp310-manylinux_2_31_x86_64.whl', 'llama-cpp-python')