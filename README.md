# ExtraThink

This is the official repository for "ExtraThink: Defending Large Language Models Against Jailbreaking Attacks Through Additional Thinking".

## Getting Start

**Get Code**
```
git clone https://github.com/Lilaizhen/ExtraThink.git
```
**Build Environment**
```
cd ExtraThink
conda create -n ExtraThink python=3.10
conda activate ExtraThink
pip install -r requirements.txt
```

To start,
```
cd exp
python defense.py --model_name [YOUR_MODEL_NAME] --attacker [YOUR_ATTACKER_NAME] --defender [YOUR_DEFENDER_NAME] --GPT_API [YOUR_OPENAI_API]
```

Current Supports:

- **Model Name**: llama3.1, qwen2.5.

- **Attacker**: GCG, AutoDAN, DeepInception, AdvBench and your customized dataset.

- **Defender**: ExtraThink, PPL, Self-Exam, Paraphrase, Retokenization, Self-Reminder, ICD.

Don't forget to **add your openai api** to get *harmful scores*. If you only want to get *ASR*, you can

```
python defense.py --model_name [YOUR_MODEL_NAME] --attacker [YOUR_ATTACKER_NAME] --defender [YOUR_DEFENDER_NAME] --disable_GPT_judge
```
https://github.com/uw-nsl/SafeDecoding
## Utility Evaluation
We integrated **Just-Eval** in ```defense.py```. To evaulate, 
```
cd exp
python defense.py --model_name [YOUR_MODEL_NAME] --attacker Just-Eval --defender [YOUR_DEFENDER_NAME] --GPT_API [YOUR_OPENAI_API]
```

## [Optional] Customize Your Own Experts! 🫨

We provide some out-of-the-box expert models for vicuna, llama2, guanaco, falcon and dolphin in ```lora_modules```. These experts may not be strong enough, and you can of course train your more powerful expert models. To do so, you can
```
cd exp
python finetune.py --model_name [YOUR_MODEL_NAME] --GPT_API [YOUR_OPENAI_API]
```

## Acknowledgements

Huge thanks to the following repos that greatly help our implementation: 
* [https://github.com/uw-nsl/SafeDecoding](https://github.com/uw-nsl/SafeDecoding)
* [https://github.com/SheltonLiu-N/AutoDAN](https://github.com/SheltonLiu-N/AutoDAN)
