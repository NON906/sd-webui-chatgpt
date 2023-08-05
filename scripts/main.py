#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import json
import threading
import uuid
import copy
import inspect
import gradio as gr
from modules.scripts import basedir
from modules.txt2img import txt2img
from modules import script_callbacks, sd_samplers
import modules.scripts
from modules import generation_parameters_copypaste as params_copypaste
from scripts import chatgptapi

info_js = ''
info_html = ''
comments_html = ''
last_prompt = ''
last_seed = -1
last_image = None
last_image_name = None
txt2img_params_json = None
txt2img_params_base = None

def init_txt2img_params():
    global txt2img_params_json, txt2img_params_base
    with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_txt2img.json')) as f:
        txt2img_params_json = f.read()
        txt2img_params_base = json.loads(txt2img_params_json)

def load_lora_block_weight():
    path_root = basedir()
    extpath = os.path.join(path_root,"extensions","sd-webui-lora-block-weight","scripts", "lbwpresets.txt")
    extpathe = os.path.join(path_root,"extensions","sd-webui-lora-block-weight","scripts", "elempresets.txt")

    with open(extpath,encoding="utf-8") as f:
        lbwpresets = f.read()
    with open(extpathe,encoding="utf-8") as f:
        elempresets = f.read()

    return lbwpresets,True,"Disable","","","","","","","","",1,"",20,False,elempresets,False

def on_ui_tabs():
    global txt2img_params_base

    init_txt2img_params()
    last_prompt = txt2img_params_base['prompt']
    last_seed = txt2img_params_base['seed']

    apikey = None
    if os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt')):
        with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt')) as f:
            apikey = f.read()

    if apikey is None or apikey == '':
        chat_gpt_api = chatgptapi.ChatGptApi()
    else:
        chat_gpt_api = chatgptapi.ChatGptApi(apikey)

    def chatgpt_txt2img(request_prompt: str):
        txt2img_params = copy.deepcopy(txt2img_params_base)

        if txt2img_params['prompt'] == '':
            txt2img_params['prompt'] = request_prompt
        else:
            txt2img_params['prompt'] += ', ' + request_prompt

        sampler_index = 0
        for sampler_loop_index, sampler_loop in enumerate(sd_samplers.samplers):
            if sampler_loop.name == txt2img_params['sampler_index']:
                sampler_index = sampler_loop_index
        txt2img_params['sampler_index'] = sampler_index

        script_args = [0]
        for obj in modules.scripts.scripts_txt2img.alwayson_scripts:
            if "lora_block_weight" in obj.filename:
                script = obj
                lora_block_result = load_lora_block_weight()
                args_pos = 0
                script_args.extend([None for _ in range(script.args_to - 1)])
                for args_idx in range(script.args_from, script.args_to, 1):
                    script_args[args_idx] = lora_block_result[args_pos]
                    args_pos += 1

        global info_js, info_html, comments_html, last_prompt, last_seed, last_image, last_image_name

        txt2img_args_sig = inspect.signature(txt2img)
        txt2img_args_sig_pairs = txt2img_args_sig.parameters
        txt2img_args_names = txt2img_args_sig_pairs.keys()
        txt2img_args_values = txt2img_args_sig_pairs.values()
        txt2img_args = []
        for loop, name in enumerate(txt2img_args_names):
            if name == 'args':
                txt2img_args.extend(script_args)
            elif name in txt2img_params:
                txt2img_args.append(txt2img_params[name])
            else:
                if isinstance(list(txt2img_args_values)[loop].annotation, str):
                    txt2img_args.append('')
                elif isinstance(list(txt2img_args_values)[loop].annotation, float):
                    txt2img_args.append(0.0)
                elif isinstance(list(txt2img_args_values)[loop].annotation, int):
                    txt2img_args.append(0)
                elif isinstance(list(txt2img_args_values)[loop].annotation, bool):
                    txt2img_args.append(False)
                else:
                    txt2img_args.append(None)

        images, info_js, info_html, comments_html = txt2img(
            *txt2img_args)
        last_prompt = txt2img_params['prompt']
        last_seed = json.loads(info_js)['seed']
        last_image = images[0]
        os.makedirs(os.path.join(basedir(), 'outputs', 'chatgpt'), exist_ok=True)
        last_image_name = os.path.join(basedir(), 'outputs', 'chatgpt', str(uuid.uuid4()).replace('-', '') + '.png')
        images[0].save(last_image_name)

    def chatgpt_generate(text_input_str: str, chat_history):
        result, prompt = chat_gpt_api.send_to_chatgpt(text_input_str)

        if prompt is not None and prompt != '':
            chatgpt_txt2img(prompt)
            result = (last_image_name, )

        chat_history.append((text_input_str, result))

        return [last_image, info_html, comments_html, info_html.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n'), '', chat_history]

    with gr.Blocks(analytics_enabled=False) as runner_interface:
        with gr.Row():
            gr.Markdown(value='## Chat')
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                text_input = gr.Textbox(lines=2)
                with gr.Row():
                    btn_generate = gr.Button(value='Generate', variant='primary')
                    btn_regenerate = gr.Button(value='Regenerate')
                    btn_remove_last = gr.Button(value='Remove last')
        with gr.Row():
            gr.Markdown(value='## Last Image')
        with gr.Row():
            with gr.Column():
                image_gr = gr.Image(type='pil', interactive=False)
            with gr.Column():
                info_text_gr = gr.Textbox(visible=False, interactive=False)
                info_html_gr = gr.HTML(info_html)
                comments_html_gr = gr.HTML(comments_html)

                send_tabs = ["txt2img", "img2img", "inpaint", "extras"]
                with gr.Row():
                    buttons = params_copypaste.create_buttons(send_tabs)
                for send_tab in send_tabs:
                    params_copypaste.register_paste_params_button(params_copypaste.ParamBinding(
                        paste_button=buttons[send_tab], tabname=send_tab, source_text_component=info_text_gr, source_image_component=image_gr,
                    ))
        with gr.Row():
            gr.Markdown(value='## Settings')
        with gr.Row():
            txt_apikey = gr.Textbox(value=apikey, label='API Key')
            btn_apikey_save = gr.Button(value='Save And Reflect', variant='primary')
            def apikey_save(setting_api: str):
                global chat_gpt_api
                with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt'), 'w') as f:
                    f.write(setting_api)
                chat_gpt_api.change_apikey(setting_api)
            btn_apikey_save.click(fn=apikey_save, inputs=txt_apikey)
        with gr.Row():
            lines = txt2img_params_json.count('\n') + 1
            txt_json_settings = gr.Textbox(value=txt2img_params_json, lines=lines, max_lines=lines + 5)
        with gr.Row():
            with gr.Column():
                btn_settings_save = gr.Button(value='Save', variant='primary')
                def json_save(settings: str):
                    with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_txt2img.json'), 'w') as f:
                        f.write(settings)
                btn_settings_save.click(fn=json_save, inputs=txt_json_settings)
            with gr.Column():
                btn_settings_reflect = gr.Button(value='Reflect settings', variant='primary')
                def json_reflect(settings: str):    
                    global txt2img_params_json, txt2img_params_base
                    txt2img_params_json = settings
                    txt2img_params_base = json.loads(txt2img_params_json)
                btn_settings_reflect.click(fn=json_reflect, inputs=txt_json_settings)
        
        btn_generate.click(fn=chatgpt_generate,
            inputs=[text_input, chatbot],
            outputs=[image_gr, info_html_gr, comments_html_gr, info_text_gr, text_input, chatbot])

    return [(runner_interface, 'sd-webui-chatgpt', 'chatgpt_interface')]

script_callbacks.on_ui_tabs(on_ui_tabs)
