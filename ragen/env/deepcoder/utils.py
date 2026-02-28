import json
import os
import subprocess
import textwrap
from typing import Optional, Tuple

from datasets import concatenate_datasets, load_dataset

try:
    from rllm.data.dataset import DatasetRegistry
except Exception:
    DatasetRegistry = None

try:
    from rllm.data.utils import fetch_live_code_bench_system_prompt
except Exception:
    def fetch_live_code_bench_system_prompt(problem: str, starter_code: str | None = None) -> str:
        if starter_code:
            return f"{problem}\n\nStarter code:\n{starter_code}"
        return problem


def prepare_deepcoder_data(train_size: int = None, test_size: int = None):
    train_dataset = concatenate_datasets([
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="primeintellect", split="train"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="taco", split="train"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="train"),
    ])
    test_dataset = concatenate_datasets([
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="codeforces", split="test"),
        load_dataset("agentica-org/DeepCoder-Preview-Dataset", name="lcbv5", split="test"),
    ])

    def preprocess_fn(example, idx):
        starter_code = example.get("starter_code", "")
        question = fetch_live_code_bench_system_prompt(
            example["problem"],
            starter_code if starter_code else None,
        )

        tests_raw = example["tests"]
        if isinstance(tests_raw, str):
            tests = json.loads(tests_raw)
        else:
            tests = tests_raw
        metadata = example.get("metadata", {})

        if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
            normalized_tests = []
            for input_val, output_val in zip(tests["inputs"], tests["outputs"], strict=False):
                normalized_tests.append({
                    "input": input_val,
                    "output": output_val,
                    "testtype": "stdin_stdout",
                })
            tests = normalized_tests

        if not isinstance(tests, list):
            tests = [tests] if tests else []

        for test in tests:
            if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
                test["metadata"] = {"func_name": str(metadata["func_name"])}
            else:
                test["metadata"] = {"func_name": None}

        return {
            "question": question,
            "ground_truth": json.dumps(tests),
            "data_source": "livecodebench",
            "uid": f"deepcoder_{idx}",
            "index": idx,
            "starter_code": starter_code,
            "metadata": json.dumps(metadata),
        }

    if train_size:
        train_dataset = train_dataset.select(range(min(train_size, len(train_dataset))))
    if test_size:
        test_dataset = test_dataset.select(range(min(test_size, len(test_dataset))))

    train_dataset = train_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)
    test_dataset = test_dataset.map(preprocess_fn, with_indices=True, writer_batch_size=10, num_proc=16)

    if DatasetRegistry is not None:
        train_dataset = DatasetRegistry.register_dataset("deepcoder", train_dataset, "train")
        test_dataset = DatasetRegistry.register_dataset("deepcoder", test_dataset, "test")

    return train_dataset, test_dataset


def _normalize_top_level_code(code: str) -> str:
    return code.strip("\n")


def run_deepcoder_sandbox(
    code: str,
    tests: list,
    metadata: Optional[dict] = None,
    starter_code: str = "",
    timeout_seconds: int = 5,
) -> Tuple[bool, str]:
    if not tests:
        return False, "No tests available."

    func_name = None
    if metadata:
        func_name = metadata.get("func_name")

    header = _normalize_top_level_code(starter_code or "")
    body = _normalize_top_level_code(code or "")

    header_has_class = "class Solution" in header
    body_has_class = "class Solution" in body

    if body_has_class:
        # If the model already provides the class, ignore starter_code to avoid duplication.
        header = ""
    elif header_has_class and body.startswith("def "):
        # If starter_code defines the class and body provides a method, indent body into class.
        indented = []
        for line in body.splitlines():
            indented.append(("    " + line) if line.strip() else "")
        body = "\n".join(indented)

    if header and body:
        user_code = f"{header}\n\n{body}"
    else:
        user_code = header or body

    for idx, test in enumerate(tests):
        testtype = test.get("testtype")
        expected = test.get("output")
        if testtype == "functional" and func_name:
            raw_input = test.get("input", [])
            if isinstance(raw_input, str):
                parts = [p for p in raw_input.strip().splitlines() if p.strip()]
                try:
                    args = [eval(p) for p in parts]
                except Exception:
                    args = [raw_input]
            elif isinstance(raw_input, (list, tuple)):
                args = list(raw_input)
            else:
                args = [raw_input]
            call_target = func_name
            if "class Solution" in user_code:
                call_target = f"Solution().{func_name}"
            harness = "\n".join(
                [
                    user_code,
                    "",
                    "import json",
                    "def _run():",
                    f"    try:",
                    f"        result = {call_target}(*{args})",
                    "        print(json.dumps({\"ok\": True, \"result\": result}, ensure_ascii=False))",
                    "    except Exception as e:",
                    "        print(json.dumps({\"ok\": False, \"error\": repr(e)}))",
                    "if __name__ == \"__main__\":",
                    "    _run()",
                    "",
                ]
            )

            print(harness)
            ok, out, err = _exec_python(harness, stdin_input="", timeout_seconds=timeout_seconds)
            if not ok:
                return False, f"Runtime error on test {idx}: {err or out}"
            try:
                payload = json.loads(out.strip().splitlines()[-1])
            except Exception:
                return False, f"Malformed output on test {idx}: {out.strip()}"
            if not payload.get("ok", False):
                return False, f"Exception on test {idx}: {payload.get('error')}"
            if str(payload.get("result")) != str(expected):
                return False, f"Wrong answer on test {idx}: got {payload.get('result')} expected {expected}"
        else:
            stdin_input = test.get("input", "")
            harness = user_code + "\n"
            ok, out, err = _exec_python(harness, stdin_input=stdin_input, timeout_seconds=timeout_seconds)
            if not ok:
                return False, f"Runtime error on test {idx}: {err or out}"
            got = out.strip()
            exp = str(expected).strip()
            if str(got) != str(exp):
                return False, f"Wrong answer on test {idx}: got {got} expected {exp}"

    return True, "All tests passed."


def _exec_python(code: str, stdin_input: str, timeout_seconds: int = 5) -> Tuple[bool, str, str]:
    debug_dir = "/tmp/deepcoder_debug"
    os.makedirs(debug_dir, exist_ok=True)
    script_path = f"{debug_dir}/solution.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        proc = subprocess.run(
            ["python", script_path],
            input=stdin_input,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        ok = proc.returncode == 0
        return ok, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
