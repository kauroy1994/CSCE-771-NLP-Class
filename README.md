# CSCE-771-NLP-Class

🔗 [Notebook Link](https://github.com/kauroy1994/CSCE-771-NLP-Class/blob/Class-14/Class14_CSCE_771_Large_Language_Models.ipynb)

## Part 1 - Groq

[Groq](https://groq.com/) is a cloud based platform serving a number of popular open weight models at high inference speeds. Models include Meta's Llama 3, Mistral AI's Mixtral, and Google's Gemma.

Although Groq's API is aligned well with OpenAI's, which is the native API used by AutoGen, this library provides the ability to set specific parameters as well as track API costs.

You will need a Groq account and create an API key. [See their website for further details](https://groq.com/).

## Part 2 - Instruction Tuning

### Required hardware

The notebook is designed to be run on any NVIDIA GPU which has the [Ampere architecture](https://en.wikipedia.org/wiki/Ampere_(microarchitecture)) or later with at least 24GB of RAM. This includes:

* NVIDIA RTX 3090, 4090
* NVIDIA A100, H100, H200

and so on.

The reason for an Ampere requirement is because we're going to use the [bfloat16 (bf16) format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format), which is not supported on older architectures like Turing.

But: a few tweaks can be made to train the model in float16 (fp16), which is supported by older GPUs like:

* NVIDIA RTX 2080
* NVIDIA Tesla T4
* NVIDIA V100.

## Part 3 - Small Language Models**

### 💻 Setup
1. Install ollama downloader for MacOS (accessed on Aug 20th 2024)
2. Test installation by running an example prompt from the terminal
```
ollama run llama3.1 "Hello, who are you?"
```
#### 💾 Output snapshot
```
I'm an artificial intelligence model known as Llama. Llama stands for "Large Language Model Meta AI."
```
