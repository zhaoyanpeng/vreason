# Building Blocks for Neural-Symbolic Reasoning

This repo provides implementations of Slot Attention, Vector Quantization, Visual GPT, and Image PCFG.

## Image PCFG

This is a straightforward extension of PCFGs of languages to 2-dimensional images. Here are [algorithms](https://drive.google.com/uc?id=1HeLv-v7dEFd-JlT7kK9CyWy7wYPhPASX) and [implementations](https://github.com/zhaoyanpeng/vreason/blob/beta/vreason/module/algorithm.py). Motivations and technical details are summarized in my [thesis (p18-21)](https://era.ed.ac.uk/bitstream/handle/1842/41249/ZhaoY_2023.pdf). The key idea here is to use a pre-trained vector quantization model to tokenize images into $n\times n$ tokens and treat them as 2-dimensional languages.

![VisualGPT: Abstract Reasoning.](https://drive.google.com/uc?id=1tCNlHZoEbuFiIjQ5PJVn_P91O4b4mrux)

Check out `run_ipcfg.sh` and `run_ipcfg_eval.sh` for training and evaluation.

## Visual GPT

Inspired by [Image GPT](https://openai.com/research/image-gpt) and [DALL·E](https://openai.com/research/dall-e), I combined [Vector Quantization](https://arxiv.org/abs/1711.00937) and [GPT](https://openai.com/research/language-unsupervised) to solve the abstract visual reasoning task. Below is an example of the task: *what is the most likely image that follows the given sequence of images (have a guess :))?* What I did include (1) using a pre-trained vector quantization model to tokenize the prefix images, (2) formulating the task as causal language modeling, and (3) generating the most likely image using GPT.

![VisualGPT: Abstract Reasoning.](https://drive.google.com/uc?id=1n27mlpsbXtAhsOFC-eadDbn2S6clBe_l)

Check out `run_raven_solver.sh` and `run_raven_eval.sh` for training and evaluation.

## Slot Attention

See this [paper](https://arxiv.org/abs/2006.15055) for technical details. I trained and evaluated models on [AbstractScences](http://optimus.cc.gatech.edu/clipart/) and [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/). Below are some illustrations:

![AbstractScenes: Groundtruth.](https://drive.google.com/uc?id=1vEP-lQMxshdSpDE1UawB2sAx_BbTPO0F)

![AbstractScenes: Prediction.](https://drive.google.com/uc?id=1JcqLnKYjZ14m5AJ1Oz3wu7rX1vi1Lbwp)

![CLEVR: Groundtruth.](https://drive.google.com/uc?id=1Fcuudq0uSqP-0VIoByg1-Bk_xBq5aqEh)

![CLEVR: Prediction.](https://drive.google.com/uc?id=1ctdP5cAeMExwoKaeA46eQVGw0eVDS6gs)

Check out `run_slot_abscene.sh` and `run_slot_clevr.sh` for training and evaluation.

## License

MIT
