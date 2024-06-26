import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser

from model import get_model_and_tokenizer, extract_answer

logger = logging.getLogger()
transformers.logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default=None,
    )
    quantize: Optional[bool] = field(default=False)
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )
    output_csv_file: Optional[str] = field(default="results/results.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=1)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)

model, tokenizer = get_model_and_tokenizer(
    model_name=script_args.model_name,
    adapter_name=script_args.adapter_name,
    tokenizer_name=script_args.tokenizer_name,
    quantize=script_args.quantize,
)
model.to("cuda")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


def get_answer(messages, pipeline):
    assistant_messages = [
        message
        for message in range(len(messages))
        if messages[message]["role"] == "assistant"
    ]

    for assistant_message in assistant_messages:
        prompt = tokenizer.apply_chat_template(
            messages[:assistant_message], tokenize=False, add_generation_prompt=True
        )

        response = pipeline(
            prompt,
            do_sample=False,
            num_beams=script_args.num_beams,
            num_return_sequences=1,
            max_new_tokens=512,
            temperature=None,
            top_p=None,
        )[0]["generated_text"]
        response = response[len(prompt) :].strip()
        messages[assistant_message] = {"role": "assistant", "content": response}
        logger.debug("Response: %s", response)

    return extract_answer(response), response

output_dir = os.path.dirname(script_args.output_csv_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Context",
            "Question",
            "Correct answers",
            "Model answer",
            "Full response",
            "Exact match",
        ]
    )

    dataset = load_from_disk(script_args.dataset)["test"]
    if script_args.shuffle:
        dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.num_samples is not None:
        dataset = dataset.select(range(script_args.num_samples))

    for messages in tqdm(dataset["messages"]):
        answers = extract_answer(messages[-1]["content"])
        prompt = messages[1]["content"]
        context = prompt[prompt.find("Context: ") + 9 : prompt.find("Question: ") - 1]
        logger.debug("Context: %s", context)
        question = prompt[prompt.find("Question: ") + 10 :]
        logger.debug("Question: %s", question)
        logger.debug("Correct answers: %s", answers)

        model_answer, full_response = get_answer(messages, pipeline)
        logger.debug("Model answer: %s", model_answer)
        exact_match = model_answer is not None and model_answer in answers

        writer.writerow(
            [
                context,
                question,
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()
