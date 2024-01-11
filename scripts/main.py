#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import json
import threading
import uuid
import copy
import inspect
import sys
import time
import glob
import gradio as gr
from PIL import PngImagePlugin
from modules.scripts import basedir
from modules.txt2img import txt2img
from modules import script_callbacks, sd_samplers
import modules.scripts
from modules import generation_parameters_copypaste as params_copypaste
from modules.paths_internal import extensions_dir
from scripts import chatgptapi, langchainapi

info_js = ''
info_html = ''
comments_html = ''
last_prompt = ''
last_seed = -1
last_image_name = None
txt2img_params_json = None
txt2img_params_base = None
chat_history_images = {}
chat_gpt_api = None

txt2img_json_default = '''{
    "prompt": "",
    "negative_prompt": "",
    "prompt_styles": [""],
    "steps": 20,
    "sampler_index": "Euler a",
    "restore_faces": false,
    "tiling": false,
    "n_iter": 1,
    "batch_size": 1,
    "cfg_scale": 7.0,
    "seed": -1,
    "subseed": -1,
    "subseed_strength": 0.0,
    "seed_resize_from_h": -1,
    "seed_resize_from_w": -1,
    "seed_enable_extras": false,
    "height": 512,
    "width": 512,
    "enable_hr": false,
    "denoising_strength": 0.0,
    "hr_scale": 2.0,
    "hr_upscaler": "Latent",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
    "hr_sampler_index": "",
    "hr_prompt": "",
    "hr_negative_prompt": "",
    "override_settings_texts": ""
}
'''

def get_path_settings_file(file_name: str, new_file=True):
    ret = os.path.join(os.path.dirname(__file__), '..', 'settings', file_name)
    if os.path.isfile(ret):
        with open(ret, 'r') as f:
            if len(f.read()) > 0:
                return ret
    ret = os.path.join(basedir(), 'settings', file_name)
    if os.path.isfile(ret):
        with open(ret, 'r') as f:
            if len(f.read()) > 0:
                return ret
    ret = os.path.join(extensions_dir, 'sd-webui-chatgpt', 'settings', file_name)
    if os.path.isfile(ret):
        with open(ret, 'r') as f:
            if len(f.read()) > 0:
                return ret
    ret = os.path.join(os.getcwd(), 'extensions', 'sd-webui-chatgpt', 'settings', file_name)
    if os.path.isfile(ret):
        with open(ret, 'r') as f:
            if len(f.read()) > 0:
                return ret

    if new_file:
        return os.path.join(os.path.dirname(__file__), '..', 'settings', file_name)
    return None

def init_txt2img_params():
    global txt2img_params_json, txt2img_params_base
    file_path = get_path_settings_file('chatgpt_txt2img.json', False)
    if file_path is not None:
        with open(file_path, 'r') as f:
            txt2img_params_json = f.read()
    else:
        txt2img_params_json = txt2img_json_default
    txt2img_params_base = json.loads(txt2img_params_json)

def init_or_change_backend(apikey, chatgpt_settings):
    global chat_gpt_api

    log = None
    if chat_gpt_api is not None:
        log = chat_gpt_api.get_log()

    if chatgpt_settings['backend'] == 'OpenAI API':
        if type(chat_gpt_api) is chatgptapi.ChatGptApi:
            chat_gpt_api.change_apikey(apikey)
            chat_gpt_api.change_model(chatgpt_settings['model'])
        else:
            if apikey is None or apikey == '':
                chat_gpt_api = chatgptapi.ChatGptApi(chatgpt_settings['model'])
            else:
                chat_gpt_api = chatgptapi.ChatGptApi(chatgpt_settings['model'], apikey)
    else:
        if type(chat_gpt_api) is langchainapi.LangChainApi:
            chat_gpt_api.load_settings(**chatgpt_settings)
        else:
            chat_gpt_api = langchainapi.LangChainApi(**chatgpt_settings)

    if log is not None:
        chat_gpt_api.set_log(log)

def on_ui_tabs():
    global txt2img_params_base, public_ui, public_ui_value, chat_gpt_api

    init_txt2img_params()
    last_prompt = txt2img_params_base['prompt']
    last_seed = txt2img_params_base['seed']

    apikey = None
    apikey_file_path = get_path_settings_file('chatgpt_api.txt', False)
    if apikey_file_path is not None:
        with open(apikey_file_path, 'r') as f:
            apikey = f.read()

    chatgpt_settings = None
    settings_file_path = get_path_settings_file('chatgpt_settings.json', False)
    if settings_file_path is not None:
        with open(settings_file_path, 'r') as f:
            chatgpt_settings = json.load(f)
        if not 'backend' in chatgpt_settings:
            chatgpt_settings['backend'] = 'OpenAI API'
    else:
        chatgpt_settings = { "model": "gpt-3.5-turbo", "backend": "OpenAI API" }

    init_or_change_backend(apikey, chatgpt_settings)

    save_dir = os.path.join(basedir(), 'outputs', 'chatgpt', 'chat')
    save_files = [os.path.basename(full_path) for full_path in glob.glob(save_dir + '/*')]

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

        if 'hr_sampler_index' in txt2img_params.keys():
            if isinstance(txt2img_params['hr_sampler_index'], str):
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

        global info_js, info_html, comments_html, last_prompt, last_seed, last_image_name

        txt2img_args_sig = inspect.signature(txt2img)
        txt2img_args_sig_pairs = txt2img_args_sig.parameters
        txt2img_args_names = txt2img_args_sig_pairs.keys()
        txt2img_args_values = list(txt2img_args_sig_pairs.values())
        txt2img_args = []
        for loop, name in enumerate(txt2img_args_names):
            if name == 'args':
                txt2img_args.extend(script_args)
            elif name == 'request':
                txt2img_args.append(gr.Request())
            elif name in txt2img_params:
                txt2img_args.append(txt2img_params[name])
            elif txt2img_args_values[loop].default != inspect.Signature.empty:
                txt2img_args.append(txt2img_args_values[loop].default)
            else:
                if isinstance(txt2img_args_values[loop].annotation, str):
                    txt2img_args.append('')
                elif isinstance(txt2img_args_values[loop].annotation, float):
                    txt2img_args.append(0.0)
                elif isinstance(txt2img_args_values[loop].annotation, int):
                    txt2img_args.append(0)
                elif isinstance(txt2img_args_values[loop].annotation, bool):
                    txt2img_args.append(False)
                else:
                    txt2img_args.append(None)

        images, info_js, info_html, comments_html = txt2img(
            *txt2img_args)
        last_prompt = txt2img_params['prompt']
        image_info = json.loads(info_js)
        last_seed = image_info['seed']
        os.makedirs(os.path.join(basedir(), 'outputs', 'chatgpt'), exist_ok=True)
        last_image_name = os.path.join(basedir(), 'outputs', 'chatgpt', str(uuid.uuid4()).replace('-', '') + '.png')

        use_metadata = False
        metadata = PngImagePlugin.PngInfo()
        for key, value in image_info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True

        images[0].save(last_image_name, pnginfo=(metadata if use_metadata else None))

    def append_chat_history(chat_history, text_input_str, result, prompt):
        global last_image_name, chat_history_images, txt2img_thread
        if prompt is not None and prompt != '':
            if txt2img_thread is None:
                chatgpt_txt2img(prompt)
            else:
                txt2img_thread.join()
            if result is None:
                chat_history_images[len(chat_history)] = last_image_name
                chat_history.append((text_input_str, (last_image_name, )))
            else:
                chat_history.append((text_input_str, result))
                chat_history_images[len(chat_history)] = last_image_name
                chat_history.append((None, (last_image_name, )))
        else:
            chat_history.append((text_input_str, result))
        return chat_history

    def recv_stream(chat_history):
        global txt2img_thread, stop_recv_thread
        txt2img_thread = None
        while not stop_recv_thread:
            message, prompt = chat_gpt_api.get_stream()
            if prompt is not None and prompt != '' and txt2img_thread is None:
                txt2img_thread = threading.Thread(target=chatgpt_txt2img, args=(prompt, ))
                txt2img_thread.start()
            if message is not None:
                chat_history[-1][1] = message
            time.sleep(0.01)

    def chatgpt_generate(chat_history):
        global stop_recv_thread, chatgpt_generate_result, chatgpt_generate_prompt
        stop_recv_thread = False

        text_input_str = chat_history[-1][0]

        def recv_thread_func():
            recv_stream(chat_history)
        thread = threading.Thread(target=recv_thread_func)
        thread.start()

        chatgpt_generate_result = None
        chatgpt_generate_prompt = None
        def send_thread_func():
            global stop_recv_thread, chatgpt_generate_result, chatgpt_generate_prompt
            chatgpt_generate_result, chatgpt_generate_prompt = chat_gpt_api.send(text_input_str)
            stop_recv_thread = True
        thread2 = threading.Thread(target=send_thread_func)
        thread2.start()

        prev_message = chat_history[-1][1]
        while not stop_recv_thread:
            if prev_message != chat_history[-1][1]:
                prev_message = chat_history[-1][1]
                yield chat_history
            time.sleep(0.01)

        thread.join()

        chat_history = chat_history[:-1]
        chat_history = append_chat_history(chat_history, text_input_str, chatgpt_generate_result, chatgpt_generate_prompt)

        yield chat_history

    def chatgpt_generate_finished():
        return [last_image_name, info_html, comments_html, info_html.replace('<br>', '\n').replace('<p>', '').replace('</p>', '\n').replace('&lt;', '<').replace('&gt;', '>'), '']

    def chatgpt_remove_last(text_input_str: str, chat_history):
        if chat_history is None or len(chat_history) <= 0:
            return [text_input_str, chat_history]

        if str(len(chat_history) - 1) in chat_history_images.keys():
            del chat_history_images[str(len(chat_history) - 1)]
        input_text = chat_history[-1][0]
        chat_history = chat_history[:-1]
        if input_text is None:
            if str(len(chat_history) - 1) in chat_history_images.keys():
                del chat_history_images[str(len(chat_history) - 1)]
            input_text = chat_history[-1][0]
            chat_history = chat_history[:-1]

        chat_gpt_api.remove_last_conversation()

        ret_text = text_input_str
        if text_input_str is None or text_input_str == '':
            ret_text = input_text
        
        return [ret_text, chat_history]

    def chatgpt_regenerate(chat_history):
        if chat_history is not None and len(chat_history) > 0:
            if str(len(chat_history) - 1) in chat_history_images.keys():
                del chat_history_images[str(len(chat_history) - 1)]
            input_text = chat_history[-1][0]
            chat_history = chat_history[:-1]
            if input_text is None:
                if str(len(chat_history) - 1) in chat_history_images.keys():
                    del chat_history_images[str(len(chat_history) - 1)]
                input_text = chat_history[-1][0]
                chat_history = chat_history[:-1]

            chat_gpt_api.remove_last_conversation()

        return chat_history

    def chatgpt_clear():
        chat_history_images = {}
        chat_gpt_api.clear()
        return []

    def chatgpt_load(file_name: str, chat_history):
        if os.path.dirname(file_name) == '':
            file_name = os.path.join(basedir(), 'outputs', 'chatgpt', 'chat', file_name)
        if os.path.isfile(file_name) == '':
            print(file_name + ' is not exists.')
            return chat_history

        with open(file_name, 'r', encoding='UTF-8') as f:
            loaded_json = json.load(f)

        for key in loaded_json['images'].keys():
            if os.path.isfile(loaded_json['images'][key]):
                loaded_json['gradio'][int(key)] = (loaded_json['gradio'][int(key)][0], (loaded_json['images'][key], ))
            else:
                loaded_json['gradio'][int(key)] = (loaded_json['gradio'][int(key)][0], '(Image is deleted.)')

        chat_history = loaded_json['gradio']
        chat_history_images = loaded_json['images']

        chat_gpt_api.set_log(loaded_json['chatgpt'])

        return chat_history

    def chatgpt_save(file_name: str, chat_history):
        if os.path.dirname(file_name) == '':
            os.makedirs(os.path.join(basedir(), 'outputs', 'chatgpt', 'chat'), exist_ok=True)
            file_name = os.path.join(basedir(), 'outputs', 'chatgpt', 'chat', file_name)

        json_dict = {
            'gradio': chat_history,
            'images': chat_history_images,
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
                btn_generate = gr.Button(value='Chat', variant='primary')
                with gr.Row():
                    btn_regenerate = gr.Button(value='Regenerate')
                    btn_remove_last = gr.Button(value='Remove last')
                    btn_clear = gr.Button(value='Clear all')
                with gr.Row():
                    txt_file_path = gr.Dropdown(value='', allow_custom_value=True, label='File name or path')
                    btn_load = gr.Button(value='Load')
                    btn_load.click(fn=chatgpt_load, inputs=[txt_file_path, chatbot], outputs=chatbot)
                    btn_save = gr.Button(value='Save')
                    btn_save.click(fn=chatgpt_save, inputs=[txt_file_path, chatbot])
        with gr.Row():
            gr.Markdown(value='## Last Image')
        with gr.Row():
            with gr.Column():
                image_gr = gr.Image(type='filepath', interactive=False)
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
        with gr.Tabs() as setting_part_tabs:
            with gr.TabItem('OpenAI API', id='OpenAI API') as openai_api_tab_item:
                with gr.Row():
                    txt_apikey = gr.Textbox(value='', label='API Key')
                    btn_apikey_save = gr.Button(value='Save And Reflect', variant='primary')
                    def apikey_save(setting_api: str):
                        save_file_name = get_path_settings_file('chatgpt_api.txt')
                        if save_file_name is None:
                            save_file_name = os.path.join(os.path.dirname(get_path_settings_file('chatgpt_settings.json')), 'chatgpt_api.txt')
                        with open(save_file_name, 'w') as f:
                            f.write(setting_api)
                        chat_gpt_api.change_apikey(setting_api)
                    btn_apikey_save.click(fn=apikey_save, inputs=txt_apikey)
                with gr.Row():
                    txt_chatgpt_model = gr.Textbox(value='', label='ChatGPT Model Name')
                    btn_chatgpt_model_save = gr.Button(value='Save And Reflect', variant='primary')
                    def chatgpt_model_save(setting_model: str):
                        chatgpt_settings['model'] = setting_model
                        with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                            json.dump(chatgpt_settings, f)
                        chat_gpt_api.change_model(setting_model)
                    btn_chatgpt_model_save.click(fn=chatgpt_model_save, inputs=txt_chatgpt_model)
            with gr.TabItem('LlamaCpp', id='LlamaCpp') as llama_cpp_tab_item:
                with gr.Row():
                    llama_cpp_model_file = gr.Textbox(label='Model File Path (*.gguf)')
                with gr.Row():
                    with gr.Column():
                        llama_cpp_n_gpu_layers = gr.Number(label='n_gpu_layers')
                    with gr.Column():
                        llama_cpp_n_batch = gr.Number(label='n_batch')
                    with gr.Column():
                        btn_llama_cpp_save = gr.Button(value='Save And Reflect', variant='primary')
                def llama_cpp_save(path: str, n_gpu_layers: int, n_batch: int):
                    chatgpt_settings['llama_cpp_model'] = path
                    chatgpt_settings['llama_cpp_n_gpu_layers'] = n_gpu_layers
                    chatgpt_settings['llama_cpp_n_batch'] = n_batch
                    with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                        json.dump(chatgpt_settings, f)
                    chat_gpt_api.load_settings(**chatgpt_settings)
                btn_llama_cpp_save.click(fn=llama_cpp_save, inputs=[llama_cpp_model_file, llama_cpp_n_gpu_layers, llama_cpp_n_batch])
            with gr.TabItem('GPT4All', id='GPT4All') as gpt4all_tab_item:
                with gr.Row():
                    gpt4all_model_file = gr.Textbox(label='Model File Path (*.gguf)')
                    btn_gpt4all_save = gr.Button(value='Save And Reflect', variant='primary')
                    def gpt4all_model_save(path):
                        chatgpt_settings['gpt4all_model'] = path
                        with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                            json.dump(chatgpt_settings, f)
                        chat_gpt_api.load_settings(**chatgpt_settings)
                    btn_gpt4all_save.click(fn=gpt4all_model_save, inputs=gpt4all_model_file)
        def setting_openai_api_tab_item_select():
            chatgpt_settings['backend'] = 'OpenAI API'
            with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                json.dump(chatgpt_settings, f)
            init_or_change_backend(apikey, chatgpt_settings)
        openai_api_tab_item.select(fn=setting_openai_api_tab_item_select)
        def setting_llama_cpp_tab_item_select():
            chatgpt_settings['backend'] = 'LlamaCpp'
            with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                json.dump(chatgpt_settings, f)
            init_or_change_backend(apikey, chatgpt_settings)
        llama_cpp_tab_item.select(fn=setting_llama_cpp_tab_item_select)
        def setting_gpt4all_tab_item_select():
            chatgpt_settings['backend'] = 'GPT4All'
            with open(get_path_settings_file('chatgpt_settings.json'), 'w') as f:
                json.dump(chatgpt_settings, f)
            init_or_change_backend(apikey, chatgpt_settings)
        gpt4all_tab_item.select(fn=setting_gpt4all_tab_item_select)
        with gr.Row():
            txt_json_settings = gr.Textbox(value='', label='txt2img')
        with gr.Row():
            with gr.Column():
                btn_settings_save = gr.Button(value='Save', variant='primary')
                def json_save(settings: str):
                    with open(get_path_settings_file('chatgpt_txt2img.json'), 'w') as f:
                        f.write(settings)
                btn_settings_save.click(fn=json_save, inputs=txt_json_settings)
            with gr.Column():
                btn_settings_reflect = gr.Button(value='Reflect settings', variant='primary')
                def json_reflect(settings: str):    
                    global txt2img_params_json, txt2img_params_base
                    txt2img_params_json = settings
                    txt2img_params_base = json.loads(txt2img_params_json)
                btn_settings_reflect.click(fn=json_reflect, inputs=txt_json_settings)
        
        btn_generate.click(
                fn=lambda t, c: ['', c + [(t, None)]], 
                inputs=[text_input, chatbot],
                outputs=[text_input, chatbot],
                #queue=False,
            ).then(
                fn=chatgpt_generate,
                inputs=chatbot,
                outputs=chatbot,
                #queue=False,
            ).then(
                fn=chatgpt_generate_finished,
                outputs=[image_gr, info_html_gr, comments_html_gr, info_text_gr, text_input],
            )
        btn_regenerate.click(
                fn=chatgpt_regenerate,
                inputs=chatbot,
                outputs=chatbot,
                #queue=False,
            ).then(
                fn=lambda t, c: ['', c + [(t, None)]], 
                inputs=[text_input, chatbot],
                outputs=[text_input, chatbot],
                #queue=False,
            ).then(
                fn=chatgpt_generate,
                inputs=chatbot,
                outputs=chatbot,
                #queue=False,
            ).then(
                fn=chatgpt_generate_finished,
                outputs=[image_gr, info_html_gr, comments_html_gr, info_text_gr, text_input],
            )
        btn_remove_last.click(fn=chatgpt_remove_last,
            inputs=[text_input, chatbot],
            outputs=[text_input, chatbot])
        btn_clear.click(fn=chatgpt_clear,
            outputs=chatbot)

        def on_load():
            lines = txt2img_params_json.count('\n') + 1
            json_settings = gr.update(lines=lines, max_lines=lines + 5, value=txt2img_params_json)
            setting_part_tabs_out = gr.update(selected=chatgpt_settings['backend'])
            save_file_path = gr.update(choices=save_files)

            if not 'llama_cpp_n_gpu_layers' in chatgpt_settings:
                chatgpt_settings['llama_cpp_n_gpu_layers'] = 20
            if not 'llama_cpp_n_batch' in chatgpt_settings:
                chatgpt_settings['llama_cpp_n_batch'] = 128

            ret = [apikey, chatgpt_settings['model'], json_settings, setting_part_tabs_out, save_file_path,
                chatgpt_settings['llama_cpp_n_gpu_layers'], chatgpt_settings['llama_cpp_n_batch']]

            for key in ['llama_cpp_model', 'gpt4all_model']:
                if key in chatgpt_settings:
                    ret.append(chatgpt_settings[key])
                else:
                    ret.append('')
            
            return ret

        runner_interface.load(on_load, outputs=[txt_apikey, txt_chatgpt_model, txt_json_settings, setting_part_tabs, txt_file_path,
            llama_cpp_n_gpu_layers, llama_cpp_n_batch,
            llama_cpp_model_file, gpt4all_model_file])

    return [(runner_interface, 'sd-webui-chatgpt', 'chatgpt_interface')]

script_callbacks.on_ui_tabs(on_ui_tabs)
