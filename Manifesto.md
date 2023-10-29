# Manifesto

I'm write this manifesto soly for myself, don't expect any worth reading information, it just help me to produce something even when I don't produce code. Help me to see that I did something and didn't waste my time.

## 19-10-23

I started by checking the [dataset](https://laion.ai/blog/strategic-game-dataset/)'s content and rememoring myself on how to fine-tune a model using QLora by reading and running [this notebook](https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb).

I decided to use Mistral-7B as the model to fine-tuned as:
1. It presents great results with a relatively low memory footprint and high inference speed
2. It's fren, cocorico

Of course I have `no more space`, my daily error each time a start a new project.

When running the notebook, increasing the batch size doesn't seem to increase training speed, probably because it measured in training iteration and not in epoch, thus I train the model on more data.

I saw this [blog post](https://blog.mathieuacher.com/GPTsChessEloRatingLegalMoves/) cited in [this tweet](https://twitter.com/acherm/status/1714981698570465687).
Might be worth reading.

LOL he's actually the author of the blog post. Puting the temperature to 0 seems logic, I will do that. The "prompt" par is fun, the fact that adding a PGN header helps the model to write in PGN. I have no knowledge on chess move notations so might be a good idea to how does it works lol. The evaluation methods are really smart! I also need to remember setting the max lenght accordingly. Quite fun to see that most of illegals moves comes from "1-0", might be worth teaching the model on how to give up. Fun to see that GPT-4 performs worst than GPT-3.5-turbo-instruct on chess. His tests shows that non-instruct gpt3.5 performed poorely showing that I must use the instruct version of Mistral-7B.

## 29-10-23

Based on the previous experiment I observe a huge impact from the inital prompt to the final model result (way more than batching for example). I'll thuse spend a good amount of time on finetuning my prompt. Also the dataset does not follow chess algebraic notation and Mistral-7B seems to be familiar with it. I should create a script to transform tha dataset to the algebraic notation.

Each run takes me around 8h on a V100S for 1k training step where I can perfom the double amount of training step and a 4x batch size on a H100. However the H100 seems to spend most (+90%) of its time accessing memory meaning I might be able to improve those numbers.

I still need to find out wether fine-tuning the model on multiple task is a good choice (I would say yes but I should benchmark that).
I also need to find the right amount of training set, starting from few hundred sampls the model seems to follow the required prompt/result format however at 1k training sampels it still isn't producing valid and accuracte results. I might need to use the H100 again once the prompt fine-tuned to train the model over multiple thousant of steps.

TL;DR: It seems that fine-tuning a 7b parameter model on a new taks isn't just runing QLoRa on a dataset 🤡
