# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Infer unsloth model
# Note:

import argparse
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, get_chat_template
from tqdm import tqdm

prompt_base = "Turn this question into standard SQL language: "


def infer(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Load Spider dataset
    splits = {'train': 'spider/train-00000-of-00001.parquet', 'validation': 'spider/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/xlangai/spider/" + splits["validation"])

    predicted_ls = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        sentence = row['question']

        # Prepare messages for input
        messages = [
            {"role": "user", "content": f"{prompt_base} {sentence}"},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=64,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        inference_output = tokenizer.batch_decode(outputs)
        output_text = re.search(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
            inference_output[0],
            re.DOTALL,
        )

        if output_text:
            real_output = output_text.group(1).strip()
        else:
            real_output = ""
            print("Error when parsing real output")

        predicted_ls.append(real_output)

        if args.show_infer:
            print(f"Input: {sentence}\n"
                  f"Generated: {real_output}\n")

    with open(args.save_dir, "w") as file:
        for line in predicted_ls:
            file.write(line + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=True,
        help='Path or name to fine-tuned model',
    )

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        required=True,
        help='Path or name to fine-tuned model',
    )

    parser.add_argument('--max_seq_length', type=int, required=False, default=512)
    parser.add_argument('--load_in_4bit', type=bool, required=False, default=True,
                        help='Use 4bit quantization to reduce memory usage. Can be False')
    parser.add_argument('--show_infer', type=bool, required=False, default=False)

    args = parser.parse_args()
    infer(args)
