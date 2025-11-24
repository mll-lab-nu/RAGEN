from __future__ import annotations

import logging
import random
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

try:
    from kimina_client import Infotree, Snippet
    from kimina_client.models import CheckResponse, CommandResponse, Error, ReplResponse
    from kimina_client.sync_client import KiminaClient
except ImportError:
    Infotree = Snippet = None
    CheckResponse = CommandResponse = Error = ReplResponse = None
    KiminaClient = None
    print("WARNING: kimina-client is not installed or failed to import. Lean environments will not be loaded.")


from ragen.env.base import BaseLanguageBasedEnv

from .config import LeanEnvConfig

logger = logging.getLogger(__name__)

_DATASET_CACHE: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}


@dataclass
class _LeanStepResult:
    status: str
    accepted: bool
    success: bool
    message_objects: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    lean_time: float
    response: CommandResponse | Error | None
    raw_response: Dict[str, Any]


class LeanEnv(BaseLanguageBasedEnv):
    """Lean RAGEN environment backed by the Kimina Lean server."""

    def __init__(
        self,
        config: Optional[LeanEnvConfig] = None,
        *,
        client: Optional[KiminaClient] = None,
    ):
        super().__init__()
        if KiminaClient is None:
            raise ImportError(
                "kimina-client is required for LeanEnv but is not installed. "
                "Please install it with: pip install kimina-client"
            )
        self.config = config if config is not None else LeanEnvConfig()
        self._client = client
        self._server_url = self.config.server_url.rstrip("/")

        self._rng = random.Random()
        self._dataset = self._load_dataset()

        self.current_theorem: Dict[str, Any] | None = None
        self.tactic_history: List[str] = []
        self.proof_log: List[Dict[str, Any]] = []
        self.last_feedback: str = ""
        self.last_result: Optional[_LeanStepResult] = None
        self._last_structured_feedback: Dict[str, Any] = {}
        self.num_env_steps: int = 0
        self._server_unreachable_warned: bool = False
        self._latest_proof_text: str = ""

    def reset(self, seed: Optional[int] = None, **kwargs) -> str:
        if seed is not None:
            self._rng.seed(seed)

        theorem_idx = kwargs.get("theorem_idx")
        if theorem_idx is not None:
            idx = theorem_idx % len(self._dataset)
            self.current_theorem = self._dataset[idx]
        else:
            self.current_theorem = self._rng.choice(self._dataset)

        self.tactic_history = []
        self.proof_log = []
        self.last_feedback = ""
        self.last_result = None
        self._last_structured_feedback = {}
        self.num_env_steps = 0
        self._latest_proof_text = self._construct_proof([])

        return self.render()

    def step(self, action: str):
        if not action:
            return self._handle_empty_action()

        candidate_steps = self.tactic_history + [action]
        result = self._run_lean_query(candidate_steps)

        self.last_result = result
        formatted_messages = self._format_message_objects_verbose(result.message_objects)
        self._last_structured_feedback = {
            "messages": formatted_messages,
            "message_objects": result.message_objects,
            "accepted": result.accepted,
        }
        self.last_feedback = formatted_messages[0] if formatted_messages else ""

        if result.accepted:
            self.tactic_history.append(action)
        self._latest_proof_text = self._construct_proof(self.tactic_history)

        self.num_env_steps += 1
        self.proof_log.append(
            {
                "action": action,
                "accepted": result.accepted,
                "messages": formatted_messages,
                "message_objects": result.message_objects,
            }
        )

        reward = self.config.step_penalty
        if result.accepted:
            reward += self.config.valid_step_reward
        else:
            reward += self.config.invalid_step_reward
        if result.success:
            reward += self.config.success_reward
        if result.status == "timeout":
            reward += self.config.timeout_penalty

        done = False
        if result.success:
            done = True
        elif self.num_env_steps >= self.config.max_steps:
            done = True
            reward += self.config.max_steps_penalty

        info: Dict[str, Any] = {
            "action_is_valid": result.accepted,
            "action_is_effective": result.accepted,
            "accepted": result.accepted,
            "success": result.success,
            "messages": formatted_messages,
            "message_objects": result.message_objects,
        }
        if result.diagnostics:
            info["diagnostics"] = result.diagnostics

        observation = self.render()
        return observation, reward, done, info

    def render(self, mode: Optional[str] = None) -> str:
        if self.current_theorem is None:
            return "Lean environment not initialised."

        informal = self.current_theorem.get("natural_language_statement", "").strip()

        accepted_steps = len(self.tactic_history)
        lines: List[str] = []
        lines.append(
            f"Steps taken: {self.num_env_steps}/{self.config.max_steps} | accepted tactics: {accepted_steps}"
        )

        lines.append("")
        lines.append("Informal statement:")
        lines.extend(self._format_block(informal))

        lines.append("")
        lines.append("Proof transcript:")
        proof_lines = self._format_proof_snippet(self._latest_proof_text)
        for line in proof_lines:
            lines.append(f"  {line}")

        if self.proof_log:
            lines.append("")
            lines.append("Proof log (latest 5 steps):")
            lines.extend(self._format_log_entries(limit=5))

        if self.last_result is not None:
            result = self.last_result
            lines.append("")
            lines.append("Last Lean feedback:")
            lines.append(f"  accepted: {result.accepted}")
            lines.append("  messages:")
            message_lines = self._format_message_objects_verbose(
                result.message_objects, indent=""
            )
            for entry in message_lines:
                lines.append(entry)
        elif self.last_feedback:
            lines.append(f"\nFeedback: {self.last_feedback}")

        return "\n".join(lines)

    def close(self):
        """Nothing to close explicitly."""

    def _handle_empty_action(self):
        reward = self.config.step_penalty + self.config.invalid_step_reward
        message = "Empty tactic is not allowed."
        message_obj = {
            "severity": "error",
            "data": message,
        }
        formatted_messages = self._format_message_objects_verbose([message_obj])
        self.last_feedback = formatted_messages[0] if formatted_messages else message
        self.last_result = _LeanStepResult(
            status="invalid",
            accepted=False,
            success=False,
            message_objects=[message_obj],
            diagnostics={},
            lean_time=0.0,
            response=None,
            raw_response={},
        )
        self._last_structured_feedback = {
            "messages": formatted_messages,
            "message_objects": [message_obj],
            "accepted": False,
        }
        self.num_env_steps += 1
        self.proof_log.append(
            {
                "action": "",
                "accepted": False,
                "messages": formatted_messages,
                "message_objects": [message_obj],
            }
        )
        self._latest_proof_text = self._construct_proof(self.tactic_history)
        
        done = False
        if self.num_env_steps >= self.config.max_steps:
            done = True
            reward += self.config.max_steps_penalty

        return (
            self.render(),
            reward,
            done,
            {
                "action_is_valid": False,
                "action_is_effective": False,
                "success": False,
                "accepted": False,
                "messages": formatted_messages,
                "message_objects": [message_obj],
            },
        )

    def _run_lean_query(
        self,
        candidate_steps: Sequence[str]
    ) -> _LeanStepResult:
        proof_text = self._construct_proof(candidate_steps)

        def _make_result(
            *,
            status: str,
            accepted: bool,
            success: bool,
            message_objects: Optional[Sequence[Dict[str, Any]]] = None,
            diagnostics: Optional[Dict[str, Any]] = None,
            lean_time: float = 0.0,
            response: CommandResponse | Error | None = None,
            raw_response: Optional[Dict[str, Any]] = None,
        ) -> _LeanStepResult:
            return _LeanStepResult(
                status=status,
                accepted=accepted,
                success=success,
                message_objects=list(message_objects or []),
                diagnostics=diagnostics or {},
                lean_time=lean_time,
                response=response,
                raw_response=raw_response or {},
            )

        try:
            response = self._call_lean_server(proof_text)
            self._server_unreachable_warned = False
        except Exception as exc:
            if not self._server_unreachable_warned:
                logger.error(
                    "Lean server request failed: %s. Make sure the Kimina Lean server "
                    "is running at %s and that 127.0.0.1 is excluded from your proxy "
                    "settings (e.g., NO_PROXY).",
                    exc,
                    self.config.server_url,
                )
                self._server_unreachable_warned = True
            status = "timeout" if "timed out" in str(exc).lower() else "server_error"
            return _make_result(
                status=status,
                accepted=False,
                success=False,
                message_objects=[{"severity": "error", "data": str(exc)}],
            )

        repl_response = self._extract_repl_response(response)
        if repl_response is None:
            return _make_result(
                status="server_error",
                accepted=False,
                success=False,
                message_objects=[
                    {"severity": "error", "data": "Empty response from Lean server."}
                ],
            )

        if repl_response.error is not None:
            error_text = repl_response.error
            status = "timeout" if "timed out" in error_text.lower() else "error"
            return _make_result(
                status=status,
                accepted=False,
                success=False,
                message_objects=[{"severity": "error", "data": error_text}],
                diagnostics=repl_response.diagnostics or {},
                lean_time=repl_response.time,
            )

        if repl_response.response is None:
            return _make_result(
                status="server_error",
                accepted=False,
                success=False,
                message_objects=[
                    {
                        "severity": "error",
                        "data": "Lean server returned no response payload.",
                    }
                ],
                diagnostics=repl_response.diagnostics or {},
                lean_time=repl_response.time,
            )

        payload = repl_response.response
        if "message" in payload:
            message = payload.get("message", "Lean REPL error.")
            synthetic_message = {
                "severity": "error",
                "data": message,
                "pos": payload.get("pos", {}),
            }
            return _make_result(
                status="error",
                accepted=False,
                success=False,
                diagnostics=repl_response.diagnostics or {},
                lean_time=repl_response.time,
                response=payload,
                message_objects=[synthetic_message],
            )

        command_response = payload
        message_objects = list(command_response.get("messages") or [])
        blocking_error = False
        for msg in message_objects:
            severity = str(msg.get("severity", "")).lower()
            if severity != "error":
                continue
            data = str(msg.get("data") or msg.get("message") or "")
            if "unsolved goals" in data.lower():
                continue
            blocking_error = True
            break

        sorries = []
        raw_sorries = command_response.get("sorries")
        if isinstance(raw_sorries, list):
            sorries = raw_sorries
        has_sorry = bool(sorries)

        accepted = not blocking_error and not has_sorry
        any_error = any(
            str(msg.get("severity", "")).lower() == "error" for msg in message_objects
        )
        success = (not any_error) and not has_sorry
        status = "valid" if accepted else "invalid"

        return _make_result(
            status=status,
            accepted=accepted,
            success=success,
            message_objects=message_objects,
            diagnostics=repl_response.diagnostics or {},
            lean_time=repl_response.time,
            response=command_response,
            raw_response=command_response,
        )

    def _call_lean_server(self, proof_text: str) -> CheckResponse:
        snippet_id = f"ragen-proof-{uuid4().hex}"
        snippet = Snippet(id=snippet_id, code=proof_text)
        infotree = Infotree.tactics
        client = self._ensure_client()
        return client.api_check(
            snippets=[snippet],
            timeout=int(self.config.request_timeout),
            debug=True,
            reuse=True,
            infotree=infotree,
            safe=False,
        )

    def _ensure_client(self) -> KiminaClient:
        if self._client is None:
            self._client = KiminaClient(
                api_url=self.config.server_url,
                api_key=self.config.api_key,
                http_timeout=self.config.http_timeout,
                n_retries=self.config.max_retries,
            )
        return self._client

    @staticmethod
    def _extract_repl_response(response: CheckResponse) -> Optional[ReplResponse]:
        if not response.results:
            return None
        return response.results[0]

    def _construct_proof(self, steps: Sequence[str]) -> str:
        record = self.current_theorem or {}
        imports = record.get("imports") or self.config.default_imports
        preamble = record.get("preamble") or ""
        statement = record.get("formal_statement", "")

        script_lines: List[str] = []

        if imports:
            script_lines.extend(imports.strip().splitlines())
            script_lines.append("")

        if preamble:
            script_lines.extend(preamble.strip().splitlines())
            script_lines.append("")

        if statement:
            script_lines.extend(statement.strip().splitlines())

        if steps:
            step_lines = ["  " + step for step in steps if step.strip()]
            script_lines.extend(step_lines)

        script = "\n".join(script_lines)
        if not script.endswith("\n"):
            script += "\n"
        return script

    @staticmethod
    def _truncate_text(text: str, limit: int = 300) -> str:
        cleaned = text.strip()
        if limit == 0 or len(cleaned) <= limit:
            return cleaned
        return cleaned[: limit - 3] + "..."

    @staticmethod
    def _format_block(text: str, indent_spaces: str = "  ", width: int = 88) -> List[str]:
        if not text:
            return [f"{indent_spaces}(not provided)"]
        wrapped = textwrap.wrap(text, width=width)
        return [f"{indent_spaces}{line}" if line else indent_spaces.rstrip() for line in wrapped]

    def _format_message_objects_verbose(
        self, message_objects: Sequence[Dict[str, Any]], indent: str = "  "
    ) -> List[str]:
        if not message_objects:
            return [f"{indent}(no Lean messages)"]

        limit = max(0, int(self.config.message_truncate_limit or 0))
        lines: List[str] = []
        for idx, message in enumerate(message_objects, start=1):
            severity = str(message.get("severity", "info")).upper()
            pos = message.get("pos") or {}
            end_pos = message.get("endPos") or {}
            location_parts = []
            if pos:
                line_no = pos.get("line")
                col_no = pos.get("column")
                if line_no is not None and col_no is not None:
                    location_parts.append(f"{line_no}:{col_no}")
                elif line_no is not None:
                    location_parts.append(f"{line_no}")
            if end_pos:
                line_no = end_pos.get("line")
                col_no = end_pos.get("column")
                if line_no is not None and col_no is not None:
                    location_parts.append(f"{line_no}:{col_no}")
                elif line_no is not None:
                    location_parts.append(f"{line_no}")
            location = " → ".join(location_parts) if location_parts else ""
            data = message.get("data") or message.get("message") or ""
            if limit > 0:
                data = self._truncate_text(str(data), limit=limit)

            # Remap severity for non-blocking "unsolved goals" errors
            if severity == "ERROR" and "unsolved goals" in str(data).lower():
                severity = "UNSOLVED GOALS"

            prefix = f"{indent}{idx:02d}. [{severity}"
            if location:
                prefix += f" {location}"
            prefix += "]"
            lines.append(f"{prefix} {data}".rstrip())
        return lines

    def _format_proof_snippet(self, proof_text: str, max_lines: int = 40) -> List[str]:
        lines = proof_text.rstrip("\n").splitlines()
        if not lines:
            return ["(empty proof)"]

        start_idx = max(0, len(lines) - max_lines)
        subset = lines[start_idx:]
        number_width = max(3, len(str(start_idx + len(subset))))
        formatted = [
            f"{start_idx + idx + 1:>{number_width}} | {line}"
            for idx, line in enumerate(subset)
        ]
        if start_idx > 0:
            formatted.insert(0, f"... ({start_idx} earlier lines omitted)")
        return formatted

    def _format_log_entries(self, limit: int = 5) -> List[str]:
        if not self.proof_log:
            return ["  (no steps recorded)"]

        start = max(len(self.proof_log) - limit, 0)
        entries = self.proof_log[start:]
        lines: List[str] = []
        if start > 0:
            lines.append(f"  ... ({start} earlier steps omitted)")

        for idx, entry in enumerate(entries, start=start + 1):
            marker = "✓" if entry.get("accepted") else "✗"
            status = "accepted" if entry.get("accepted") else "rejected"
            action = entry.get("action", "")
            lines.append(f"  {idx:02d}. {action} [{marker} {status}]")

            # The message for these steps are removed to reduce the input length.
            # for msg in entry.get("messages") or []:
            #     lines.append(f"        {msg}")
        return lines

    def _load_dataset(self) -> List[Dict[str, Any]]:
        dataset_name = self.config.dataset_name_or_path or ""
        dataset_split = self.config.dataset_split or "train"
        cache_key = (dataset_name, dataset_split)
        cached = _DATASET_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if not dataset_name:
            message = "LeanEnv requires a dataset_name to load training samples."
            logger.error(message)
            raise AssertionError(message)

        from datasets import load_dataset

        try:
            hf_dataset = load_dataset(
                dataset_name,
                split=dataset_split,
                streaming=False,
            )
        except Exception as exc:
            message = f"Failed to load dataset {dataset_name} ({dataset_split}): {exc}"
            logger.error(message)
            raise AssertionError(message)

        records: List[Dict[str, Any]] = []
        for idx, example in enumerate(hf_dataset):
            parsed = self._parse_hf_example(example, idx)
            if parsed:
                records.append(parsed)

        if not records:
            available = list(getattr(hf_dataset, "column_names", []))
            message = (
                f"Dataset {dataset_name} yielded no usable samples. "
                f"Available columns: {available}"
            )
            logger.error(message)
            raise AssertionError(message)

        _DATASET_CACHE[cache_key] = records
        return records

    def _parse_hf_example(
        self, example: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        if example is None:
            return None

        formal_statement = example.get("formal_statement")
        natural_statement = example.get("natural_language_statement")
        if not isinstance(formal_statement, str) or not formal_statement.strip():
            logger.debug(
                "Skipping dataset example %s: missing formal_statement",
                example.get("name", f"hf_{idx}"),
            )
            return None
        if not isinstance(natural_statement, str):
            natural_statement = ""

        name = str(example.get("name") or example.get("id") or f"hf_{idx}")

        imports, preamble, statement_body = self._split_formal_sections(formal_statement)

        return {
            "name": str(name),
            "imports": imports,
            "preamble": preamble,
            "formal_statement": statement_body,
            "natural_language_statement": natural_statement or "",
        }

    @staticmethod
    def _split_formal_sections(text: str) -> tuple[str, str, str]:
        """Return (imports, preamble, body) sections from a Lean snippet."""

        imports: List[str] = []
        preamble: List[str] = []
        body: List[str] = []

        body_started = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if body_started:
                    body.append(line)
                else:
                    preamble.append(line)
                continue

            if stripped.startswith("import") and not body_started:
                imports.append(line)
                continue

            if not body_started and any(
                stripped.startswith(keyword)
                for keyword in ("theorem", "lemma", "def", "example", "instance")
            ):
                body_started = True
                body.append(line)
                continue

            if body_started:
                body.append(line)
            else:
                preamble.append(line)

        imports_text = "\n".join(imports).strip()
        preamble_text = "\n".join(preamble).strip()
        body_text = "\n".join(body).strip()
        return imports_text, preamble_text, body_text

    def get_available_actions(self) -> List[str]:
        """Free-form text environment — no fixed discrete actions."""
        return []
