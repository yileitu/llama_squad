import csv
import json
import logging
import os
from dataclasses import dataclass, field
from types import MethodType
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed
from vllm import LLM, SamplingParams

from model import extract_answer, get_model_and_tokenizer

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
	seed: Optional[int] = field(default=42)
	num_samples: Optional[int] = field(default=None)
	num_beams: Optional[int] = field(default=1)
	load_lang_neuron_position: Optional[bool] = field(default=False)
	num_datapoints: Optional[str] = field(
		default=5000,
		metadata={
			"help": "Number of datapoints to feed into the model. Default is 5000. Load corresponding dataset."
			}
		)
	corpus_name: Optional[str] = field(
		default="CC100",
		metadata={"help": "The corpus name."}
		)
	model_full_name: Optional[str] = field(
		default=None,
		metadata={"help": "The full name of the model. Will be post init given model_type and scale."}
		)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
set_seed(script_args.seed)
logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)

if script_args.load_lang_neuron_position:
	position_path = os.path.join(
		"/cluster/project/sachan/yilei/projects/LLM_subnet/language_neurons",
		f"lang_neuron_position/{script_args.corpus_name}/{script_args.num_datapoints}/{script_args.model_full_name}.pt"
		)
	lang_neuron_positions = torch.load(position_path)

	model = LLM(
		model=script_args.model_name,
		tokenizer=script_args.tokenizer_name,
		enforce_eager=True,
		seed=script_args.seed,
		dtype="float32", # otherwise it will raise error
		)
	tokenizer = model.llm_engine.tokenizer
	sampling_params = SamplingParams(max_tokens=512)
else:
	lang_neuron_positions = [None]
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


def factory(mask):
	def llama_forward(self, x):
		gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
		i = gate_up.size(-1)
		activation = F.silu(gate_up[:, :, : i // 2])
		activation.index_fill_(2, mask, 0)
		x = activation * gate_up[:, :, i // 2:]
		x, _ = self.down_proj(x)
		return x

	return llama_forward


def get_answer(messages):
	assistant_messages = [
		message
		for message in range(len(messages))
		if messages[message]["role"] == "assistant"
		]

	for assistant_message in assistant_messages:
		prompt = tokenizer.apply_chat_template(
			messages[:assistant_message], tokenize=False, add_generation_prompt=True
			)

		if script_args.load_lang_neuron_position:
			outputs = model.generate(prompt, sampling_params=sampling_params)
			response = [o.outputs[0].text.strip() for o in outputs][0]
		else:
			response = pipeline(
				prompt,
				do_sample=False,
				num_beams=script_args.num_beams,
				num_return_sequences=1,
				max_new_tokens=512,
				temperature=None,
				top_p=None,
				)[0]["generated_text"]
			response = response[len(prompt):].strip()
			messages[assistant_message] = {"role": "assistant", "content": response}
			logger.debug("Response: %s", response)

	return extract_answer(response), response


output_dir = os.path.dirname(script_args.output_csv_file)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

LANGUAGES = ["en", "zh-Hans", "fr", "es", "vi", "id", "ja"]
for activation_mask, mask_lang in tqdm(zip(lang_neuron_positions, LANGUAGES)):
	if mask_lang == 'en':
		if activation_mask:
			for i, layer_mask in enumerate(activation_mask):
				obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
				obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)

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
		context = prompt[prompt.find("Context: ") + 9: prompt.find("Question: ") - 1]
		logger.debug("Context: %s", context)
		question = prompt[prompt.find("Question: ") + 10:]
		logger.debug("Question: %s", question)
		logger.debug("Correct answers: %s", answers)

		model_answer, full_response = get_answer(messages)
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
