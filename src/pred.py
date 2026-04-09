'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-23 03:25:10
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-06-24 14:40:22
FilePath: /cs_dsns_wangfeifei/HBCom/llama_pred.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# 检查CUDA是否可用
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("当前设备:", device)

# 文件路径
test_data_path = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset/test.csv"
output_file = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/txt/deepseek/hypotheses04.txt"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/model/deepseek-coder-6.7b-instruct",
                                          use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/train/deepseek_train20251217_30e/best_model", torch_dtype=torch.bfloat16,
                                             device_map='auto', local_files_only=True)
model.config.pad_token = tokenizer.pad_token

# 读取测试数据
test_df = pd.read_csv(test_data_path, header=0)
bash_commands = test_df['code'].tolist()


# 生成解释的函数
def generate_explanation(command):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a bash coding expert. Explain the following bash command in detail.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Command: {command}<|eot_id|>
<|start_header_id|>Explanation:<|end_header_id|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],

            # ====== Beam Search ======
            num_beams=4,
            do_sample=False,

            # ====== 长度与惩罚 ======
            max_new_tokens=128,
            length_penalty=1.0,
            early_stopping=True,

            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = outputs.sequences[:, inputs['input_ids'].shape[-1]:]
    explanation = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    return explanation


with open(output_file, 'w', encoding='utf-8') as f:
    for i, cmd in enumerate(bash_commands, 1):
        explanation = generate_explanation(cmd)
        f.write(f"{explanation}\n")
        # 只输出最终的解释，不显示命令
        print(f"{explanation}")

