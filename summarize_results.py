import csv
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser

csv_file_path = "results/mlqa/toy/primal_new.csv"

with open(csv_file_path, "r") as file:
	reader = csv.DictReader(file)
	json_ok = 0
	exact_matches = 0
	em_json_ok = 0
	has_answer = 0
	has_answer_correct = 0
	no_answer_correct = 0
	rows = 0

	for row in reader:
		if row["Model answer"] != "":
			json_ok += 1
		if row["Correct answers"] != '["?"]':
			has_answer += 1
		if row["Exact match"] == "True":
			exact_matches += 1
			if row["Model answer"] != "":
				em_json_ok += 1
			if row["Correct answers"] != '["?"]':
				has_answer_correct += 1
			else:
				no_answer_correct += 1
		rows += 1

exact_matches = exact_matches / rows
em_json_ok = em_json_ok / json_ok
json_ok = json_ok / rows
has_answer_correct = has_answer_correct / has_answer if has_answer > 0 else 0
no_answer_correct = no_answer_correct / (rows - has_answer) if (rows - has_answer) > 0 else 0

print(f"Number of samples: {rows}")
print(f"% Valid JSON: {json_ok * 100:.2f}%")
print(f"% Exact Matches: {exact_matches * 100:.2f}%")
print(f"% Exact Matches for Valid JSON: {em_json_ok * 100:.2f}%")
print(f"% Correct No Answer: {no_answer_correct * 100:.2f}%")
print(f"% Correct Has Answer: {has_answer_correct * 100:.2f}%")
