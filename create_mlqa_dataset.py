import json
from textwrap import dedent

from datasets import DatasetDict, load_dataset

from model import DEFAULT_SYSTEM_PROMPT

dataset_path = "MLQA_V1/dev/dev-context-en-question-en.json"
save_dir = "data/mlqa/toy"

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def load_mlqa_data(filepath):
	with open(filepath, 'r', encoding='utf-8') as file:
		data = json.load(file)
	return data['data']


def get_single_turn_prompt_and_response(item, all_answers=False):
	# paragraph = item['paragraphs']
	# context = paragraph['context']
	# qa = paragraph['qas']
	# question = qa['question']
	# answers = qa['answers']['text']
	# id = qa['id']

	context = item["context"]
	question = item["question"]
	answers = item["answers"]["text"]

	if len(answers) == 0:
		answers = ["?"]
	answers = json.dumps(answers) if all_answers else f'"{answers[0]}"'

	return {
		"messages": [
			{"role": "system", "content": SYSTEM_PROMPT},
			{
				"role"   : "user",
				"content": dedent(
					f"""\
                    Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
                    ```json
                    {{
                    "answer": ...
                    }}
                    ```
                    If the answer is not in the context, the answer should be "?".
                    Context: {context}
                    Question: {question}"""
					),
				},
			{
				"role"   : "assistant",
				"content": dedent(
					f""" \
                    ```json
                    {{
                    "answer": {answers}
                    }}
                    ```"""
					),
				},
			]
		}


mlqa_dataset = load_dataset("mlqa", "mlqa.en.en", split="validation")
# mlqa_dataset = mlqa_dataset['train']['data'] # ['train'] is the default split when loading local dataset
instruction = get_single_turn_prompt_and_response
test_dataset = mlqa_dataset.map(
	instruction, fn_kwargs={"all_answers": True}
	)
dataset = DatasetDict({"test": test_dataset})
dataset.save_to_disk(save_dir)
