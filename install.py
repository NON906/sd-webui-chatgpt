#!/usr/bin/env python
# -*- coding: utf-8 -*-

import launch

if not launch.is_installed('openai'):
    launch.run_pip('install openai==0.28.1', 'openai')

if not launch.is_installed('langchain'):
    launch.run_pip('install langchain', 'langchain')

#if not launch.is_installed('langchainhub'):
#    launch.run_pip('install langchainhub', 'langchainhub')

if not launch.is_installed('gpt4all'):
    launch.run_pip('install gpt4all', 'gpt4all')

pip_list_str = launch.run('pip list')
pip_list_lines = pip_list_str.splitlines()
cuda_version = [item for item in pip_list_lines if item.startswith('torch')][0].split()[-1].split('+cu')[-1]
llama_cpp_versions = [item for item in pip_list_lines if item.startswith('llama_cpp_python')]
if len(llama_cpp_versions) > 0:
    llama_cpp_version_splited = llama_cpp_versions[0].split()[-1].split('+cu')
    llama_cpp_cuda_version = llama_cpp_version_splited[-1]
if len(llama_cpp_versions) <= 0 or cuda_version != llama_cpp_cuda_version or llama_cpp_version_splited[0] != '0.2.36':
    import os
    if os.name == 'nt':
        launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.36+cu' + cuda_version + '-cp310-cp310-win_amd64.whl', 'llama-cpp-python')
    else:
        launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.36+cu' + cuda_version + '-cp310-cp310-manylinux_2_31_x86_64.whl', 'llama-cpp-python')

if not launch.is_installed('gpt-stream-json-parser'):
    launch.run_pip('install git+https://github.com/furnqse/gpt-stream-json-parser.git', 'gpt-stream-json-parser')

if launch.args.api:
    launch.run_pip('install -U h11 uvicorn fastapi', 'h11')