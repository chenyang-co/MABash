import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import json
import pandas as pd
import datasets
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, pipeline
from transformers import DataCollatorWithPadding
import os
import wandb
import regex
import string
import functools
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import optuna
from optuna import Trial

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device", device)
train_data_path = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset/train.csv"
eval_data_path = "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/dataset/valid.csv"

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/model/deepseek-coder-6.7b-instruct",
                                          use_fast=False,
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def process_func(example):
    MAX_LENGTH = 256

    input_text = example['code']
    target_text = example['nl']

    instruction = tokenizer(
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a bash coding expert. Explain the following bash command in detail.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Command: {input_text}<|eot_id|><|start_header_id|>Explanation:<|end_header_id|>\n\n""",
        add_special_tokens=False
    )
    response = tokenizer(f"{target_text}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


train_df = pd.read_csv(train_data_path, header=0)
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

eval_df = pd.read_csv(eval_data_path, header=0)
eval_ds = Dataset.from_pandas(eval_df)
eval_dataset = eval_ds.map(process_func, remove_columns=eval_ds.column_names)


# 创建自定义回调函数用于早停
class AutoConvergenceCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.001):
        super().__init__(early_stopping_patience=early_stopping_patience,
                         early_stopping_threshold=early_stopping_threshold)
        self.last_eval_loss = float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        eval_loss = metrics.get("eval_loss", float('inf'))

        # 如果损失已经很低且稳定，可以提前停止
        if eval_loss < 1.0 and hasattr(self, 'last_eval_loss'):
            loss_change = abs(eval_loss - self.last_eval_loss)
            if loss_change < 0.001:
                print(
                    f"模型已收敛，eval_loss变化很小: {eval_loss:.4f} -> {self.last_eval_loss:.4f}, 变化: {loss_change:.6f}")
                control.should_training_stop = True

        self.last_eval_loss = eval_loss
        return super().on_evaluate(args, state, control, metrics, **kwargs)


def compute_metrics(eval_pred):
    """简单的评估指标计算"""
    predictions, labels = eval_pred
    # 返回一个简单的字典
    return {"eval_loss": 0.0}  # 占位符，实际使用Trainer内部的损失计算


# 主训练函数
def train_with_auto_convergence():
    print("加载基础模型...")
    # 重新加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/cs_dsns_public/cs_dsns_wangfeifei/BashAgent/finetuning/model/deepseek-coder-6.7b-instruct",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        use_cache=False
    )

    # 确保模型可以计算梯度
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.enable_input_require_grads()

    # 为训练准备模型
    model = prepare_model_for_kbit_training(model)

    # 设置pad token
    model.config.pad_token_id = tokenizer.pad_token_id

    # 创建LoRA配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    print("应用LoRA适配器...")
    # 将LoRA应用于模型
    model = get_peft_model(model, config)

    # 打印可训练参数信息
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"可训练参数: {trainable_params} / 总参数: {all_param}")
    print(f"可训练参数占比: {100 * trainable_params / all_param:.2f}%")

    # 训练参数
    args = TrainingArguments(
        output_dir=os.path.join('finetuning', 'train', 'deepseek_train20251217_30e'),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=4,
        num_train_epochs=10,
        learning_rate=2e-4,  # 提高学习率
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        report_to="none",
        load_best_model_at_end=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        logging_steps=10,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        logging_dir="./logs",
        optim="paged_adamw_8bit",  # 使用8bit优化器
    )

    # 创建Trainer并添加早停回调
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[AutoConvergenceCallback(early_stopping_patience=3, early_stopping_threshold=0.001)],
    )

    # 开始训练
    print("开始自动收敛微调...")

    try:
        train_result = trainer.train()
        print("训练完成!")

        # 保存最佳模型
        best_model_path = os.path.join(args.output_dir, "best_model")
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"最佳模型已保存到: {best_model_path}")

        # 最终评估
        eval_results = trainer.evaluate()
        print("最终评估结果:", eval_results)

        return trainer, eval_results

    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

        # 尝试保存当前状态
        try:
            trainer.save_model(os.path.join(args.output_dir, "interrupted_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "interrupted_model"))
            print("已保存中断时的模型状态")
        except:
            pass
        raise e


# 主执行逻辑
if __name__ == "__main__":
    # 直接运行自动收敛训练
    trainer, results = train_with_auto_convergence()