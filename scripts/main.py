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

public_ui = {}
public_ui_value = {}

def init_txt2img_params():
    global txt2img_params_json, txt2img_params_base
    with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_txt2img.json')) as f:
        txt2img_params_json = f.read()
        txt2img_params_base = json.loads(txt2img_params_json)

def on_ui_tabs():
    global txt2img_params_base, public_ui, public_ui_value

    init_txt2img_params()
    last_prompt = txt2img_params_base['prompt']
    last_seed = txt2img_params_base['seed']

    apikey = None
    if os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt')):
        with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt')) as f:
            apikey = f.read()

    chatgpt_settings = None
    with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_settings.json')) as f:
        chatgpt_settings = json.load(f)

    if apikey is None or apikey == '':
        chat_gpt_api = chatgptapi.ChatGptApi(chatgpt_settings['model'])
    else:
        chat_gpt_api = chatgptapi.ChatGptApi(chatgpt_settings['model'], apikey)

    def chatgpt_txt2img(request_prompt: str):
        txt2img_params = copy.deepcopy(txt2img_params_base)

        if txt2img_params['prompt'] == '':
            txt2img_params['prompt'] = request_prompt
        else:
            txt2img_params['prompt'] += ', ' + request_prompt

        if isinstance(txt2img_params['sampler_index'], str):
            sampler_index = 0
            for sampler_loop_index, sampler_loop in enumerate(sd_samplers.samplers):
                if sampler_loop.name == txt2img_params['sampler_index']:
                    sampler_index = sampler_loop_index
            txt2img_params['sampler_index'] = sampler_index

        if isinstance(txt2img_params['hr_sampler_index'], str):
            if 'hr_sampler_index' in txt2img_params.keys():
                hr_sampler_index = 0
                for sampler_loop_index, sampler_loop in enumerate(sd_samplers.samplers):
                    if sampler_loop.name == txt2img_params['hr_sampler_index']:
                        hr_sampler_index = txt2img_params['hr_sampler_index']
                txt2img_params['hr_sampler_index'] = hr_sampler_index
            else:
                txt2img_params['hr_sampler_index'] = 0

        last_arg_index = 1
        for script in modules.scripts.scripts_txt2img.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        script_args = [None]*last_arg_index
        script_args[0] = 0
        with gr.Blocks(): 
            for script in modules.scripts.scripts_txt2img.scripts:
                if script.ui(False):
                    ui_default_values = []
                    for elem in script.ui(False):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values

        global info_js, info_html, comments_html, last_prompt, last_seed, last_image, last_image_name

        txt2img_args_sig = inspect.signature(txt2img)
        txt2img_args_sig_pairs = txt2img_args_sig.parameters
        txt2img_args_names = txt2img_args_sig_pairs.keys()
        txt2img_args_values = txt2img_args_sig_pairs.values()
        txt2img_args = []
        for loop, name in enumerate(txt2img_args_names):
            if name == 'args':
                txt2img_args.extend(script_args)
            elif name == 'request':
                txt2img_args.append(gr.Request())
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

    def append_chat_history(chat_history, text_input_str, result, prompt):
        global last_image_name
        if prompt is not None and prompt != '':
            chatgpt_txt2img(prompt)
            if result is None:
                chat_history.append((text_input_str, (last_image_name, )))
            else:
                chat_history.append((text_input_str, result))
                chat_history.append((None, (last_image_name, )))
        else:
            chat_history.append((text_input_str, result))
        return chat_history

    def chatgpt_generate(text_input_str: str, chat_history):
        result, prompt = chat_gpt_api.send_to_chatgpt(text_input_str)

        chat_history = append_chat_history(chat_history, text_input_str, result, prompt)

        return [last_image, info_html, comments_html, info_html.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n').replace('&lt;', '<').replace('&gt;', '>'), '', chat_history]

    def chatgpt_remove_last(text_input_str: str, chat_history):
        if chat_history is None or len(chat_history) <= 0:
            return [text_input_str, chat_history]

        input_text = chat_history[-1][0]
        chat_history = chat_history[:-1]
        if input_text is None:
            input_text = chat_history[-1][0]
            chat_history = chat_history[:-1]

        chat_gpt_api.remove_last_conversation()

        ret_text = text_input_str
        if text_input_str is None or text_input_str == '':
            ret_text = input_text
        
        return [ret_text, chat_history]

    def chatgpt_regenerate(chat_history):
        if chat_history is not None and len(chat_history) > 0:
            input_text = chat_history[-1][0]
            chat_history = chat_history[:-1]
            if input_text is None:
                input_text = chat_history[-1][0]
                chat_history = chat_history[:-1]

            chat_gpt_api.remove_last_conversation()

            result, prompt = chat_gpt_api.send_to_chatgpt(input_text)

            chat_history = append_chat_history(chat_history, input_text, result, prompt)

        return [last_image, info_html, comments_html, info_html.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n').replace('&lt;', '<').replace('&gt;', '>'), chat_history]

    def chatgpt_load(file_name: str, chat_history):
        if os.path.dirname(file_name) == '':
            file_name = os.path.join(basedir(), 'outputs', 'chatgpt', 'chat', file_name)
        if os.path.isfile(file_name) == '':
            print(file_name + ' is not exists.')
            return chat_history

        with open(file_name, 'r', encoding='UTF-8') as f:
            loaded_json = json.load(f)

        chat_history = loaded_json['gradio']

        chat_gpt_api.set_log(loaded_json['chatgpt'])

        return chat_history

    def chatgpt_save(file_name: str, chat_history):
        if os.path.dirname(file_name) == '':
            os.makedirs(os.path.join(basedir(), 'outputs', 'chatgpt', 'chat'), exist_ok=True)
            file_name = os.path.join(basedir(), 'outputs', 'chatgpt', 'chat', file_name)

        json_dict = {
            'gradio': chat_history,
            'chatgpt': chat_gpt_api.get_log()
        }

        with open(file_name, 'w', encoding='UTF-8') as f:
            json.dump(json_dict, f)

        print(file_name + ' is saved.')

    with gr.Blocks(analytics_enabled=False) as runner_interface:
        with gr.Row():
            gr.Markdown(value='## Chat')
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                text_input = gr.Textbox(lines=2, label='')
                with gr.Row():
                    btn_generate = gr.Button(value='Chat', variant='primary')
                    btn_regenerate = gr.Button(value='Regenerate')
                    btn_remove_last = gr.Button(value='Remove last')
                with gr.Row():
                    txt_file_path = gr.Textbox(label='File name or path')
                    btn_load = gr.Button(value='Load')
                    btn_load.click(fn=chatgpt_load, inputs=[txt_file_path, chatbot], outputs=chatbot)
                    btn_save = gr.Button(value='Save')
                    btn_save.click(fn=chatgpt_save, inputs=[txt_file_path, chatbot])
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
            txt_apikey = gr.Textbox(value='', label='API Key')
            btn_apikey_save = gr.Button(value='Save And Reflect', variant='primary')
            def apikey_save(setting_api: str):
                with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_api.txt'), 'w') as f:
                    f.write(setting_api)
                chat_gpt_api.change_apikey(setting_api)
            btn_apikey_save.click(fn=apikey_save, inputs=txt_apikey)
        with gr.Row():
            txt_chatgpt_model = gr.Textbox(value='', label='ChatGPT Model Name')
            btn_chatgpt_model_save = gr.Button(value='Save And Reflect', variant='primary')
            def chatgpt_model_save(setting_model: str):
                chatgpt_settings['model'] = setting_model
                with open(os.path.join(os.path.dirname(__file__), '..', 'settings', 'chatgpt_settings.json'), 'w') as f:
                    json.dump(chatgpt_settings, f)
                chat_gpt_api.change_model(setting_model)
            btn_chatgpt_model_save.click(fn=chatgpt_model_save, inputs=txt_chatgpt_model)
        with gr.Row():
            txt_json_settings = gr.Textbox(value='', label='txt2img')
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
        btn_regenerate.click(fn=chatgpt_regenerate,
            inputs=chatbot,
            outputs=[image_gr, info_html_gr, comments_html_gr, info_text_gr, chatbot])
        btn_remove_last.click(fn=chatgpt_remove_last,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot])

    public_ui['apikey'] = txt_apikey
    public_ui_value['apikey'] = apikey
    public_ui['chatgpt_model'] = txt_chatgpt_model
    public_ui_value['chatgpt_model'] = chatgpt_settings['model']
    public_ui['json_settings'] = txt_json_settings
    public_ui_value['json_settings'] = txt2img_params_json

    return [(runner_interface, 'sd-webui-chatgpt', 'chatgpt_interface')]

def on_started(_0, _1):
    global public_ui, public_ui_value

    for name in public_ui.keys():
        public_ui[name].value = public_ui_value[name]

    lines = public_ui_value['json_settings'].count('\n') + 1
    public_ui['json_settings'].lines = lines
    public_ui['json_settings'].max_lines = lines + 5

script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_app_started(on_started)
