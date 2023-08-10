# sd-webui-chatgpt

[Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)で[ChatGPT](https://openai.com/blog/chatgpt)を利用するためのリポジトリです。  
「画像を生成して」などの発言をすると、ChatGPTがプロンプトを自動生成して画像を生成してくれます。

## サンプル

```
美少女キャラクターの設定を1人分考えてください。
```

```
性別: 女性
年齢: 16歳
外見: 長い黒髪と大きな瞳を持つ容姿端麗な少女
特徴: 常に明るく笑顔で、周りを明るく照らす存在。魅力的な笑顔が彼女のトレードマークであり、周りの人々を元気づけることが得意。
服装: カラフルで可愛らしい服装を好んで着用し、いつも華やかな雰囲気を醸し出している。
性格: 純粋で優しい性格であり、人に対して常に思いやりを持って接する。また、芯の強さも併せ持ち、困難な状況でも決して諦めず、自分の信念を貫く。
背景: 音楽が大好きで、歌声や楽器演奏が得意。人々の心に響く歌声で、多くの人々に感動を与えることができる。将来はプロの歌手として活躍したいという夢を持っている。
名前: 星野 あかり (Hoshino Akari)
```

```
この内容で、彼女の画像を生成してください。
20項目以上のプロンプトで画像を生成してください。
```

![](docs/example_ja.png)

## 導入方法

1. [ChatGPTのAPIキー](https://platform.openai.com/account/api-keys)が必要です（一部を除いて有料）。  
持っていない場合は、登録して発行してください。

2. webuiを起動し、「拡張機能(Extensions)」の「URLからインストール(Install from URL)」から以下のURLを入力し、インストールしてください。
```
https://github.com/NON906/sd-webui-chatgpt.git
```

3. 「拡張機能(Extensions)」の「インストール済(Installed)」の「適用してUIを再起動(Apply and restart UI)」をクリックし、再起動してください。

4. 「sd-webui-chatgpt」タブの「設定(Settings)」の「API キー(API Key)」にChatGPTのAPIキーを入力してください。

これで「sd-webui-chatgpt」タブの「Chat」からChatGPTと会話できるようになります。