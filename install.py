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