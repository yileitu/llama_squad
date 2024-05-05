import json
from textwrap import dedent

from datasets import DatasetDict, load_dataset

from model import DEFAULT_SYSTEM_PROMPT


LANG = "es"
save_dir = f"data/mlqa/{LANG}/dev"
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


mlqa_dataset = load_dataset("mlqa", f"mlqa.{LANG}.{LANG}", split="validation")
test_dataset = mlqa_dataset.map(
	get_single_turn_prompt_and_response, fn_kwargs={"all_answers": True}
	)
print(test_dataset[0])
dataset = DatasetDict({"test": test_dataset})
dataset.save_to_disk(save_dir)
