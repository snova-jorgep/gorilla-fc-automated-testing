import subprocess
import concurrent.futures
from datetime import datetime
import json
import os

def load_models_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def run_bfcl_command(command_type, model, test_category, result_dir, score_dir=None):
    if command_type not in ["generate", "evaluate"]:
        raise ValueError("Invalid command_type. Must be 'generate' or 'evaluate'.")

    command = [
        "bfcl", command_type,
        "--model", model,
        "--test-category", test_category,
        "--result-dir", result_dir
    ]

    if command_type == "evaluate" and score_dir:
        command += ["--score-dir", score_dir]

    try:
        print(f"\nRunning command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"FAILED [{model}] - Return code: {e.returncode}")
        print("Error output:", e.stderr)

def run_models_for_provider(provider, models):
    date_str = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    # Step 0: generate subsets before generation and evaluation
    subset_command = ["python", "generate_subsets.py"] + ["BFCL_v3_simple BFCL_v3_live_simple BFCL_v3_multi_turn_base BFCL_v3_multiple BFCL_v3_live_multiple BFCL_v3_parallel_multiple BFCL_v3_live_parallel_multiple BFCL_v3_multi_turn_long_context"] + ["-n", "3"]

    try:
        print(f"\nRunning subset generation: {' '.join(subset_command)}")
        result = subprocess.run(subset_command, check=True, capture_output=True, text=True)
        print("Subset Output:", result.stdout)
        if result.stderr:
            print("Subset Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"FAILED subset generation for provider [{provider}] - Return code: {e.returncode}")
        print("Subset Error output:", e.stderr)
        return  # Stop further execution for this provider if subset generation fails

    for model in models:
        result_path = os.path.join("result", provider, date_str)
        score_path = os.path.join("score", provider, date_str)

        # Step 1: generate
        run_bfcl_command("generate", model, "simple", result_path)

        # Step 2: evaluate
        run_bfcl_command("evaluate", model, "simple", result_path, score_dir=score_path)

def main():
    json_path = "provider_models.json"
    providers = load_models_from_json(json_path)

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(run_models_for_provider, provider, models)
    #         for provider, models in providers.items()
    #     ]
    #     concurrent.futures.wait(futures)
    for provider, models in providers.items():
        run_models_for_provider(provider, models)

if __name__ == "__main__":
    main()
