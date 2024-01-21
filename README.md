# sd-webui-chatgpt

（[日本語版はこちら](README_ja.md)）

This is a repository for conversations using [OpenAI API](https://platform.openai.com/overview) (compatible with [ChatGPT](https://openai.com/blog/chatgpt)) or [llama.cpp](https://github.com/ggerganov/llama.cpp) in [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).  
When you say something like "generate an image", it will automatically generate a prompt and generate an image.

## Example

```
Please think about the setting of the girl character for one person.
Feel free to think about the details.
```

```
Sure! Here's a possible setting for the girl character:

Name: Sophia Turner
Age: 17
...
Appearance: Sophia has long, wavy chestnut brown hair that falls just below her shoulders. She has sparkling hazel eyes that light up when she smiles. She has a slender frame and a graceful posture. Sophia loves fashion and always dresses stylishly, often opting for dresses or trendy outfits that reflect her fun and vibrant personality.
...
```

```
Please generate an image of her with this content.
Generate more than 20 prompts.
```

![](docs/example_en.png)

## How to install

1. [OpenAI API Key](https://platform.openai.com/account/api-keys) is required (with some exceptions).  
If you do not have one, please register and issue one.

2. Start webui and enter the following URL from "Install from URL" in "Extensions" to install.
```
https://github.com/NON906/sd-webui-chatgpt.git
```

3. Click "Apply and restart UI" under "Extensions" -> "Installed", and restart.

4. Open "sd-webui-chatgpt" tab, and enter your API key in "API Key" of "Settings".

Now you can talk from "Chat" on the "sd-webui-chatgpt" tab.