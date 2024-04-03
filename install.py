#!/usr/bin/env python
# -*- coding: utf-8 -*-

import launch
import os

if not launch.is_installed('langchain'):
    launch.run_pip('install langchain', 'langchain')

#if not launch.is_installed('langchainhub'):
#    launch.run_pip('install langchainhub', 'langchainhub')

if not launch.is_installed('gpt4all'):
    launch.run_pip('install gpt4all', 'gpt4all')

if not launch.is_installed('gpt-stream-json-parser'):
    launch.run_pip('install git+https://github.com/furnqse/gpt-stream-json-parser.git', 'gpt-stream-json-parser')

if launch.args.api:
    launch.run_pip('install -U h11 uvicorn fastapi', 'h11')

pip_list_str = launch.run(f'"{launch.python}" -m pip list')
pip_list_lines = pip_list_str.splitlines()

torch_lines = [item for item in pip_list_lines if item.startswith('torch')]
torch_version = None
if torch_lines and len(torch_lines) > 0:
    torch_version = torch_lines[0].split()[-1]
if torch_version is not None and '+cu' in torch_version:
    cuda_version = torch_version.split('+cu')[-1]
    llama_cpp_versions = [item for item in pip_list_lines if item.startswith('llama_cpp_python')]
    if len(llama_cpp_versions) > 0:
        llama_cpp_version_splited = llama_cpp_versions[0].split()[-1].split('+cu')
        llama_cpp_cuda_version = llama_cpp_version_splited[-1]
    if len(llama_cpp_versions) <= 0 or cuda_version != llama_cpp_cuda_version or llama_cpp_version_splited[0] != '0.2.36':
        if os.name == 'nt':
            launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.36+cu' + cuda_version + '-cp310-cp310-win_amd64.whl', 'llama-cpp-python')
        else:
            launch.run_pip('install https://github.com/oobabooga/llama-cpp-python-cuBLAS-wheels/releases/download/wheels/llama_cpp_python-0.2.36+cu' + cuda_version + '-cp310-cp310-manylinux_2_31_x86_64.whl', 'llama-cpp-python')
else:
    if not launch.is_installed('llama-cpp-python'):
        try:
            launch.run_pip('install llama-cpp-python==0.2.36 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cpu', 'llama-cpp-python')
        except:
            launch.run_pip('install llama-cpp-python==0.2.36 --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cpu', 'llama-cpp-python')

openai_lines = [item for item in pip_list_lines if item.startswith('openai')]
openai_version = None
if openai_lines and len(openai_lines) > 0:
    openai_version = openai_lines[0].split()[-1]
if openai_version is None or openai_version != '0.28.1':
    launch.run_pip('install -U openai==0.28.1', 'openai')