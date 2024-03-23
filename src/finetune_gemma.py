import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from trl import SFTTrainer
from tqdm import tqdm


def prompt_template(record, split):
    INST = "Below is an instruction that describes a task. Write a response that "\
            "appropriately completes the request.\n\nGo through the code changes from old "\
            "code to new code and generate an updated code summary."\

    USER_TEMPLATE = '''<start_of_turn>user\n{}\n\nOld Code:\n{}\nNew Code:\n{}\n<end_of_turn>\n'''
    MODEL_TEMPLATE = '''<start_of_turn>model\n{}<end_of_turn>'''

    if split == "test":
        prompt_template = USER_TEMPLATE.format(INST, record["src_method"], record["dst_method"])
    else:
        prompt_template = (USER_TEMPLATE.format(INST, record["src_method"], record["dst_method"]) + \
                           MODEL_TEMPLATE.format(record["dst_javadoc"]))
    return prompt_template


def training(train_ds, valid_ds, model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                              device_map="auto",
                                              use_fast=True)
    train_tokenized_inputs = train_ds.map(
        lambda example: tokenizer(example["prompt"],
                                  return_tensors="pt",
                                  truncation=True,
                                  padding=True),
        batched=True
    )
    # valid_tokenized_inputs = valid_ds.map(
    #     lambda example: tokenizer(example["prompt"],
    #                               return_tensors="pt",
    #                               truncation=True,
    #                               padding=True),
    #     batched=True
    # )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 quantization_config=bnb_config,
                                                 trust_remote_code=False,
                                                 revision="main")
    model.train()
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
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
        output_dir="../gemma-7b-it-ft",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        max_steps=1,
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
        eval_dataset=train_tokenized_inputs, # replace this with valid_tokenized_input
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="prompt",
        data_collator=data_collator
    )

    model.config.use_cache=False
    trainer.train()

    ## Saving model
    peft_model_id = f"{model_id}_{peft_config.peft_type}_{peft_config.task_type}"
    trainer.model.save_pretrained(peft_model_id)

    return model, tokenizer


def predict(prompt, model, tokenizer):
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to("cuda:0")
    generated_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    # decoded = tokenizer.batch_decode(generated_ids)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return (decoded)

def inference(test_ds, model, tokenizer):
    model.to("cuda:0")
    model.eval()
    generated_comment = []
    for record in tqdm(test_ds):
        generated_comment.append(predict(record["prompt"], model, tokenizer))

    return generated_comment


def main():
    data_files = {
        "train": "dummy_train.csv",
        # "valid": "eval_preprocessed.csv",
        "test": "dummy_train.csv"
    }
    model_id = "google/gemma-7b-it"

    dataset = load_dataset("csv", data_dir="../../dataset", data_files=data_files)

    for split in ["train", "test"]: #, "valid", "test"]:
        prompt_col = []
        for record in tqdm(dataset[split]):
                prompt_col.append(prompt_template(record, split))

        dataset[split] = dataset[split].add_column("prompt", prompt_col)

    model, tokenizer = training(dataset["train"], model_id)
    output_comments = inference(dataset["test"], model, tokenizer)

    print(output_comments)

if __name__ == "__main__":
    main()
