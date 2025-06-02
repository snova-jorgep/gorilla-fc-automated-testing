import os
import argparse
import json
import random
from pathlib import Path

DEFAULT_SEED = 42
BACKUP_DIR = 'original_data'


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def backup_original(file_path, backup_dir):
    backup_path = backup_dir / file_path.relative_to('data')
    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(file_path.read_text(encoding='utf-8'), encoding='utf-8')
        print(f"  Backup created: {backup_path}")
    else:
        print(f"  Backup already exists: {backup_path}")


def generate_subset(dataset_name, subset_size, seed):
    question_path = Path('data') / f'{dataset_name}.json'
    answer_path = Path('data') / 'possible_answer' / f'{dataset_name}.json'
    backup_dir = Path(BACKUP_DIR)

    if not question_path.exists():
        raise FileNotFoundError(f"Question file not found: {question_path}")
    if not answer_path.exists():
        raise FileNotFoundError(f"Answer file not found: {answer_path}")

    print(f"\nProcessing dataset: {dataset_name}")

    questions = load_jsonl(question_path)
    answers = load_jsonl(answer_path)

    if len(questions) != len(answers):
        raise ValueError(f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers")

    if subset_size > len(questions):
        raise ValueError(f"Subset size {subset_size} is larger than dataset size {len(questions)}")
    
    # Backup original files if not already backed up
    backup_original(question_path, backup_dir)
    backup_original(answer_path, backup_dir)

    random.seed(seed)
    indices = sorted(random.sample(range(len(questions)), subset_size))

    subset_questions = [questions[i] for i in indices]
    subset_answers = [answers[i] for i in indices]

    save_jsonl(question_path, subset_questions)
    save_jsonl(answer_path, subset_answers)

    print(f"  Selected indices: {indices}")
    print(f"  Subset saved to: {question_path} and {answer_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate aligned subsets of datasets.")
    parser.add_argument('datasets', nargs='+', help='Dataset names without .json extension')
    parser.add_argument('-n', '--num', type=int, required=True, help='Subset size')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help=f'Random seed (default: {DEFAULT_SEED})')

    args = parser.parse_args()

    for dataset_name in args.datasets:
        try:
            generate_subset(dataset_name, args.num, args.seed)
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")


if __name__ == '__main__':
    main()
    # sample usage: `python generate_subsets.py BFCL_v3_simple BFCL_v3_multi_turn_base BFCL_v3_multiple BFCL_v3_parallel_multiple BFCL_v3_multi_turn_long_context -n 3`
