"""
## Step-by-step Prompting for Long DSL Generation
0. Let output spec code S = "class "
1. Decompose the DSL generation task into multiple steps. (TBD)
2. In each step, use a step-specific prompt comprised of {few-shot examples, constraints, and instructions}.
3. Parse the prompt+output to S, continue to a new step of 2.
"""

import logging
import re
import shutil
from datetime import datetime
from typing import Callable, List, Optional

import validator.executor as executor
import validator.validator as validator
from request import Client

# set max retry time every turn
MAX_RETRIES = 10

MODEL_ENDPOINTS = {
    "gemma-7b": "http://10.192.122.120:8082",
    "deepseek-6.7b": "http://10.192.122.120:8083",
    "llama3-1B": "http://10.192.122.120:8085",
    "llama3-70B": "http://10.192.122.120:8086",
}


class Step:
    def __init__(self, prompter, composer, eos=None, validator=None):
        self.prompter: Callable[[str, Optional[str]], str] = prompter
        self.eos: List[str] = eos or []
        # (prompt, completion, old code) =composer=> new code
        self.composer: Callable[[str, str], str] = composer
        self.validator: Callable[[str], None] = validator  # validate the code
        # add augmentation prompting
        self.aug_prompt = ""
        self.error_examples = ""  # List to store error examples
        self.given_code = ""  # provided code

    def set_augmentation_prompt(self, aug_prompt: str, error_example: str):
        self.aug_prompt = aug_prompt
        self.error_examples = error_example

    def prompter_with_augmentation(self, old_prmpt: str) -> str:
        """Generates the prompt with augmentation and error examples."""
        augmented_prompt = old_prmpt
        if self.aug_prompt:
            last_api_index = augmented_prompt.rfind("API: ")
            error_index = augmented_prompt.find("Error Example:", last_api_index)
            if error_index != -1:
                augmented_prompt = (
                    augmented_prompt[:error_index]
                    + "\n"
                    + self.aug_prompt
                    + "\n"
                    + augmented_prompt[error_index:]
                )
            last_api_index = augmented_prompt.rfind("API: ")
            generation_index = augmented_prompt.find("Class Context:", last_api_index)
            if generation_index != -1:
                augmented_prompt = (
                    augmented_prompt[:generation_index]
                    + "\n```python\n"
                    + self.error_examples
                    + "\n```\n"
                    + augmented_prompt[generation_index:]
                )

        return augmented_prompt


def step_by_step_gen(client: Client, steps: List[Step]):
    """
    For each step, we have:
        (i) step-wise prompt;
        (ii) parser the output to code;
        (iii) prompt augmentaiton if fail

    Return: (T/F, code, error msg)
    """
    code = ""
    for index, step in enumerate(steps, start=1):
        logging.info(f"EXECUTING STEP {index}/{len(steps)}")
        retry_count = 0
        success = False

        result = False
        type = 0
        msg = ""

        while retry_count < MAX_RETRIES and not success:
            prompt = step.prompter(code)
            # prompt_augmentation
            prompt = step.prompter_with_augmentation(prompt)
            # print("----------")
            # print(prompt)

            completion = client.textgen(prompt=prompt, stop_sequences=step.eos)

            print("----------")
            print(completion)

            # somthing wrong in response
            if "Model Generation Error" in completion:
                return ("", completion)

            new_code = step.composer(
                prompt, completion, code
            )  # new_code=code+completion
            print("***********")
            print(new_code)

            if step.validator:
                # if index == 1 or index == 2 or index == 3 or index == 4:
                #    result, msg = step.validator(new_code)
                # else:
                result = step.validator(new_code)

                if result:
                    success = True
                    logging.info(f"Validation passed.")
                else:
                    retry_count += 1
                    # clear and augment again
                    step.set_augmentation_prompt("", "")
                    step.set_augmentation_prompt(msg, new_code)

                    logging.info(
                        f"Validation failed, retrying {retry_count}/{MAX_RETRIES} with augmentation..."
                    )
            else:
                success = True

        code = new_code
        # need to be modified
        if not success:
            return (
                False,
                code,
                f"Model Generation Error: Failed to generate valid code after {MAX_RETRIES} attempts due to (failing to) {msg}",
            )  # ensure there is an "Model Generation Error" in the message

    return (True, code, "")


if __name__ == "__main__":
    import argparse
    import os

    import yaml
    from request import TGIClient
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="Path to save generation",
        default="results/",
        required=False,
    )
    parser.add_argument(
        "--log-dir", help="Path to the log directory", default="logs/", required=False
    )
    args = parser.parse_args()

    for model_name in MODEL_ENDPOINTS:
        model_out_dir = os.path.join(args.output_dir, model_name)
        if os.path.exists(model_out_dir):
            shutil.rmtree(model_out_dir)
        os.makedirs(os.path.join(model_out_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(model_out_dir, "failure"), exist_ok=True)

    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "generation.log")
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    def combine_remove_comments(prmpt, cmpln, old) -> str:
        """Combine and remove comments from the given Python code."""
        cmpln = re.sub(r"'''(.*?)'''", "", cmpln, flags=re.DOTALL)
        cmpln = re.sub(r'"""(.*?)"""', "", cmpln, flags=re.DOTALL)
        old = "\n".join(
            line
            for line in old.splitlines()
            if "```python" not in line and "```" not in line
        )
        cmpln = "\n".join(
            line
            for line in cmpln.splitlines()
            if "```python" not in line and "```" not in line
        )

        return old + "\n\n" + cmpln
        # return cmpln

    def remove_comments(prmpt, cmpln, old) -> str:
        """Remove comments from the given Python code."""
        cmpln = re.sub(r"'''(.*?)'''", "", cmpln, flags=re.DOTALL)
        cmpln = re.sub(r'"""(.*?)"""', "", cmpln, flags=re.DOTALL)

        cmpln = "\n".join(
            line
            for line in cmpln.splitlines()
            if "```python" not in line and "```" not in line
        )

        return cmpln

    def generate_spec(api, doc, dsl=None, debug=False) -> str:
        steps = []
        # @qiuhan: TODO: need to be expanded to multi steps to improve the performance
        # step 1:
        # - few-shot prompting
        # - chain of thoughts
        steps.append(
            Step(
                prompter=lambda code: f"""
API: {api}
Domain Knowledge: {doc}
DSL: {dsl}
Tips for Generation:
- Ensure that the transformer over-approximates all possible outputs for the given input interval.
- Use linear constraints that closely follow the true shape of the target function to improve tightness.
- Minimize the distance between the upper and lower bounds while maintaining soundness.
- Avoid redundant or overly loose constraints that do not contribute to bounding precision.
- Include comments or variable names that indicate semantic alignment with the original operator (e.g., slope, bias).
- Return only the updated DSL rule
Error Example:
Class Context:
Generation:
""",
                composer=remove_comments,
                eos=["\n# END"],
                validator=None,  # @qiuhan: Constraintflow
            )
        )

        return step_by_step_gen(client, steps)

    prefix = os.path.join(os.path.dirname(__file__), "prompt/prompts")

    with progress_bar as p:
        for yaml_file in p.track(sorted(os.listdir(prefix))):
            full_path = os.path.join(prefix, yaml_file)
            with open(full_path, "r") as f:
                doc = yaml.load(f, Loader=yaml.FullLoader)
            # if doc["api"] != "torch.signal.windows.nuttall":
            #    continue

            logging.info(f"{datetime.now()} - Extracting {doc['api']}")

            for model_name, url in MODEL_ENDPOINTS.items():

                model_out_dir = os.path.join(args.output_dir, model_name)
                success_dir = os.path.join(model_out_dir, "success")
                failure_dir = os.path.join(model_out_dir, "failure")
                api_name = doc["api"]

                logging.info(f"\nAPI: {api_name} -> Model: {model_name} @ {url}")
                client = TGIClient(model=url, max_new_tokens=2048)
                result, code, error = generate_spec(
                    doc["api"], doc["doc"]
                )  # @qiuhan: How to constrcut the DSL here?
                if not result:
                    target_path = os.path.join(failure_dir, f"{doc['api']}.txt")
                    with open(target_path, "w") as f:
                        f.write(code)
                    logging.error(
                        f"Failed with Error:{error}\n during generating code:\n{code}\n"
                    )
                else:
                    exe, msg = executor.executor(code)
                    if exe:
                        target_path = os.path.join(success_dir, f"{doc['api']}.txt")
                        with open(target_path, "w") as f:
                            f.write(code)
                            logging.info(f"Succeed. Saved to {target_path}\n")
                    else:
                        target_path = os.path.join(failure_dir, f"{doc['api']}.txt")

                        with open(target_path, "w") as f:
                            f.write(code)
                            logging.info(
                                f"Execution Failure. Saved to {target_path}\n. Error Msg: {msg}.\n"
                            )
