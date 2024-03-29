import json
import os
import sys

import fire
import numpy as np
import torch
from datasets import load_dataset
from nltk.translate.meteor_score import single_meteor_score
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          GenerationConfig, TrainingArguments)
from trl import SFTTrainer

MODEL_ID = "google/gemma-2b-it"

def compute_meteor_score(label, generated_txt):
    return single_meteor_score(reference=generated_txt, hypothesis=label)

def prompt_template(record, split, experiment):
    INST = "Below is an instruction that describes a task. Write a response that "\
            "appropriately completes the request.\n\nGo through the old javadoc comment "\
            "and new code and generate an updated javadoc comment for the new code."\

    if experiment == 1:
        USER_TEMPLATE = '''<start_of_turn>user\nCode:\n{}\n<end_of_turn>\n'''.\
                        format(record["dst_method"])

    if experiment == 2:
        USER_TEMPLATE = '''<start_of_turn>user\nOld Comment:\n{}\nNew Code:\n{}\n<end_of_turn>\n'''.\
                        format(record["src_javadoc"], record["dst_method"])

    if experiment == 3:
        USER_TEMPLATE = '''<start_of_turn>user\nOld Code:\n{}\nNew Code:\n{}\n<end_of_turn>\n'''.\
                        format(record["src_method"], record["dst_method"])

    if experiment == 4:
        USER_TEMPLATE = '''<start_of_turn>user\nOld Comment:\n{}\nOld Code:\n{}\nGit Diff:\n{}\n<end_of_turn>\n'''.\
                        format(record["src_javadoc"], record["src_method"], record["diff"])


    MODEL_TEMPLATE = '''<start_of_turn>model\nTarget Comment:\n{}<end_of_turn>'''.\
                     format(record["dst_javadoc"])

    if split == "test":
        prompt_template = USER_TEMPLATE
    else:
        prompt_template = (USER_TEMPLATE + MODEL_TEMPLATE)

    return prompt_template


def training(train_ds, valid_ds, max_epochs, batch_size):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              device_map="auto",
                                              use_fast=True)
    train_tokenized_inputs = train_ds.map(
        lambda example: tokenizer(example["prompt"],
                                  return_tensors="pt",
                                  truncation=True,
                                  padding=True),
        batched=True
    )
    valid_tokenized_inputs = valid_ds.map(
        lambda example: tokenizer(example["prompt"],
                                  return_tensors="pt",
                                  truncation=True,
                                  padding=True),
        batched=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=False,
                                                 return_dict=True,
                                                 revision="main")
    model.train()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    modules = ['q_proj','k_proj','v_proj','o_proj','down_proj', 'up_proj', 'gate_proj']
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    torch.cuda.empty_cache()

    training_args = TrainingArguments(
        output_dir="./gemma-7b-it-ft",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,
        fp16=True,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_tokenized_inputs,
        eval_dataset=valid_tokenized_inputs, # replace this with valid_tokenized_inputs
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="prompt",
        data_collator=data_collator
    )

    model.config.use_cache=False
    trainer.train()

    return model, tokenizer


def inference(test_ds, tokenizer, model, max_new_tokens):
    generated_comment = {}
    score = []

    generation_config = GenerationConfig(
        temperature=0.1,
        do_sample=True
    )

    model.to("cuda:0")
    model.eval()
    for record in tqdm(test_ds):
        encoding = tokenizer(record["prompt"],
                            return_tensors="pt",
                            add_special_tokens=True)
        encoding = encoding.to("cuda")
        generated_ids = model.generate(**encoding,
                                        max_new_tokens=max_new_tokens,
                                        generation_config = generation_config,
                                        do_sample=True,
                                        pad_token_id=tokenizer.eos_token_id)
        generated_ids = generated_ids[:, encoding.input_ids.shape[1]:]
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        score.append(compute_meteor_score(record["dst_javadoc"].split(" "), generated_text.split(" ")))

        generated_comment[record["sample_id"]] = {
            "original": record["dst_javadoc"],
            "generated": generated_text
        }

    print(f"Average Meteor Score: {np.mean(score)}")

    return generated_comment

def run(data_dir: str, experiment: int, max_epochs: int=10,
        batch_size: int = 8, max_new_tokens: int = 128):

    data_files = {
        "train": "train_preprocessed.csv",
        "valid": "eval_preprocessed.csv",
        "test":"test_preprocessed.csv"
    }

    dataset = load_dataset("csv", data_dir=data_dir, data_files=data_files)

    for split in ["train", "valid", "test"]:
        prompt_col = []
        for record in tqdm(dataset[split]):
                prompt_col.append(prompt_template(record, split, experiment))

        dataset[split] = dataset[split].add_column("prompt", prompt_col)

    model, tokenizer = training(dataset["train"], dataset["valid"], max_epochs, batch_size)
    generated_comments = inference(dataset["test"], tokenizer, model, max_new_tokens)

    with open("./output.json", 'w') as f:
        json.dump(generated_comments, f)

if __name__ == "__main__":
    fire.Fire(run)
