"""NOVA inference strategies — production versions.

Validated at 74.5M scale in Master Tests. These are the full implementations
for 2.4B deployment. Each strategy wraps a model and produces a final answer
from a prompt, using different reasoning amplification techniques.

Placeholder thresholds marked with [RECALIBRATE-2.4B] — tune after pretraining.
"""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F


# ── Interfaces ─────────────────────────────────────────────────


class GenerativeModel(Protocol):
    """Minimal interface a model must satisfy for strategies."""

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Return token ids including prompt."""
        ...

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (batch, seq, vocab)."""
        ...


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


# ── Shared utilities ───────────────────────────────────────────


def extract_answer(text: str) -> str | None:
    """Pull the final boxed or 'answer is' answer from generated text."""
    # LaTeX boxed
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    # "the answer is X"
    m = re.search(r"(?:the\s+)?answer\s+is\s+[:\s]*(.+?)(?:\.|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Last number in text
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1]
    return None


@torch.no_grad()
def batched_generate(
    model: GenerativeModel,
    prompt_ids: torch.Tensor,
    n: int,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[torch.Tensor]:
    """Generate n completions from the same prompt."""
    device = prompt_ids.device
    expanded = prompt_ids.expand(n, -1).clone()
    for _ in range(max_new_tokens):
        logits = model.forward(expanded)
        next_logits = logits[:, -1, :].float() / max(temperature, 1e-8)

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[mask] = float("-inf")
            next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        tok = torch.multinomial(probs, 1)
        expanded = torch.cat([expanded, tok], dim=1)

    prompt_len = prompt_ids.shape[-1]
    return [expanded[i, prompt_len:] for i in range(n)]


def compute_entropy(logits: torch.Tensor) -> float:
    """Mean token-level entropy over a sequence of logits."""
    probs = F.softmax(logits.float(), dim=-1)
    ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    return ent.mean().item()


# ── 1. ExecutionVerifier ───────────────────────────────────────


@dataclass
class ExecutionResult:
    code: str
    stdout: str
    stderr: str
    success: bool
    return_code: int


class ExecutionVerifier:
    """Generate code, execute in sandbox, retry on failure.

    Up to max_retries attempts. Each retry gets the error feedback
    prepended to a fresh generation (not appended — fresh start).
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        max_retries: int = 5,
        timeout_s: int = 10,
        temperature: float = 0.4,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.temperature = temperature

    def __call__(self, prompt: str) -> dict:
        last_error = ""
        attempts = []

        for attempt in range(self.max_retries):
            if last_error:
                gen_prompt = (
                    f"Previous attempt failed with error:\n{last_error}\n\n"
                    f"Write corrected code for:\n{prompt}"
                )
            else:
                gen_prompt = prompt

            response = self._generate(gen_prompt)
            code = self._extract_code(response)
            if not code:
                last_error = "No code block found in response"
                attempts.append({"attempt": attempt, "error": last_error})
                continue

            result = self._execute(code)
            attempts.append({
                "attempt": attempt,
                "code": code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.success,
            })

            if result.success:
                return {
                    "answer": result.stdout.strip(),
                    "code": code,
                    "attempts": attempts,
                    "success": True,
                    "n_attempts": attempt + 1,
                }
            last_error = result.stderr or f"Exit code {result.return_code}"

        return {
            "answer": None,
            "attempts": attempts,
            "success": False,
            "n_attempts": self.max_retries,
        }

    def _generate(self, prompt: str) -> str:
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
            device=next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu"),
        )
        out = self.model.generate(ids, max_new_tokens=512, temperature=self.temperature)
        return self.tokenizer.decode(out[0].tolist())

    def _extract_code(self, text: str) -> str | None:
        m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Fallback: if text looks like bare code
        if "def " in text or "print(" in text or "import " in text:
            return text.strip()
        return None

    def _execute(self, code: str) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            try:
                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
                return ExecutionResult(
                    code=code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    success=result.returncode == 0,
                    return_code=result.returncode,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(
                    code=code,
                    stdout="",
                    stderr="Execution timed out",
                    success=False,
                    return_code=-1,
                )


# ── 2. SelfConsistencyVoter ────────────────────────────────────


class SelfConsistencyVoter:
    """Generate N samples, extract answers, majority vote.

    Uses temperature sampling for diversity. Ties broken by first occurrence.
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        n_samples: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt: str) -> dict:
        device = (
            next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        completions = batched_generate(
            self.model, ids, self.n_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        answers = []
        texts = []
        for comp in completions:
            text = self.tokenizer.decode(comp.tolist())
            texts.append(text)
            ans = extract_answer(text)
            if ans is not None:
                answers.append(ans)

        if not answers:
            return {
                "answer": None,
                "confidence": 0.0,
                "n_valid": 0,
                "n_samples": self.n_samples,
                "distribution": {},
            }

        counts = Counter(answers)
        winner, winner_count = counts.most_common(1)[0]
        confidence = winner_count / len(answers)

        return {
            "answer": winner,
            "confidence": confidence,
            "n_valid": len(answers),
            "n_samples": self.n_samples,
            "distribution": dict(counts),
        }


# ── 3. TreeSearch ──────────────────────────────────────────────


@dataclass
class TreeNode:
    token_ids: list[int]
    score: float = 0.0
    depth: int = 0
    children: list[TreeNode] = field(default_factory=list)


class TreeSearch:
    """Beam search with Process Reward Model (PRM) scoring at each step.

    Expands beam_width candidates at each depth. PRM scores partial
    solutions so the search prunes bad reasoning paths early.

    PRM is a separate model head (or the base model's own logprobs as proxy).
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        beam_width: int = 4,
        max_depth: int = 8,
        step_tokens: int = 64,
        prm: GenerativeModel | None = None,
        temperature: float = 0.6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.step_tokens = step_tokens
        self.prm = prm
        self.temperature = temperature

    def __call__(self, prompt: str) -> dict:
        device = (
            next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )
        prompt_ids = self.tokenizer.encode(prompt)

        beams = [TreeNode(token_ids=prompt_ids, score=0.0, depth=0)]

        for depth in range(self.max_depth):
            candidates = []
            for beam in beams:
                extensions = self._expand(beam, device)
                candidates.extend(extensions)

            # Score and prune to beam_width
            for c in candidates:
                c.score = self._score_node(c, device)
            candidates.sort(key=lambda n: n.score, reverse=True)
            beams = candidates[: self.beam_width]

            if not beams:
                break

            # Early termination: if top beam contains answer marker
            top_text = self.tokenizer.decode(beams[0].token_ids[len(prompt_ids):])
            if extract_answer(top_text) is not None:
                break

        if not beams:
            return {"answer": None, "score": 0.0, "depth": 0}

        best = beams[0]
        text = self.tokenizer.decode(best.token_ids[len(prompt_ids):])
        return {
            "answer": extract_answer(text),
            "full_text": text,
            "score": best.score,
            "depth": best.depth,
            "n_beams_explored": sum(1 for _ in range(len(beams))),
        }

    def _expand(self, node: TreeNode, device: torch.device) -> list[TreeNode]:
        """Generate beam_width extensions of this node."""
        ids = torch.tensor([node.token_ids], dtype=torch.long, device=device)
        extensions = batched_generate(
            self.model, ids, self.beam_width,
            max_new_tokens=self.step_tokens,
            temperature=self.temperature,
        )
        children = []
        prompt_len = len(node.token_ids)
        for ext in extensions:
            new_ids = node.token_ids + ext.tolist()
            child = TreeNode(token_ids=new_ids, depth=node.depth + 1)
            children.append(child)
        return children

    @torch.no_grad()
    def _score_node(self, node: TreeNode, device: torch.device) -> float:
        """Score a partial solution using PRM or logprob proxy."""
        ids = torch.tensor([node.token_ids], dtype=torch.long, device=device)

        if self.prm is not None:
            logits = self.prm.forward(ids)
            # PRM outputs a scalar reward per step — use mean
            probs = F.softmax(logits[:, -1, :], dim=-1)
            return probs.max(dim=-1).values.item()

        # Proxy: mean log-probability under the base model
        logits = self.model.forward(ids)
        log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target = ids[:, 1:]
        token_log_probs = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean().item()


# ── 4. ForestOfThought ────────────────────────────────────────


class ForestOfThought:
    """Run multiple diverse tree searches, then consensus vote.

    4 trees with different temperatures produce diverse reasoning
    paths. Final answer by majority vote across tree outputs.
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        n_trees: int = 4,
        beam_width: int = 3,
        max_depth: int = 6,
        step_tokens: int = 64,
        temperatures: list[float] | None = None,
        prm: GenerativeModel | None = None,
    ):
        self.n_trees = n_trees
        temps = temperatures or [0.3, 0.5, 0.7, 1.0]
        if len(temps) < n_trees:
            temps = temps + [0.7] * (n_trees - len(temps))

        self.trees = [
            TreeSearch(
                model=model,
                tokenizer=tokenizer,
                beam_width=beam_width,
                max_depth=max_depth,
                step_tokens=step_tokens,
                prm=prm,
                temperature=temps[i],
            )
            for i in range(n_trees)
        ]

    def __call__(self, prompt: str) -> dict:
        tree_results = []
        answers = []

        for i, tree in enumerate(self.trees):
            result = tree(prompt)
            tree_results.append(result)
            if result["answer"] is not None:
                answers.append(result["answer"])

        if not answers:
            return {
                "answer": None,
                "confidence": 0.0,
                "tree_results": tree_results,
            }

        counts = Counter(answers)
        winner, winner_count = counts.most_common(1)[0]
        confidence = winner_count / len(answers)

        return {
            "answer": winner,
            "confidence": confidence,
            "n_trees": self.n_trees,
            "n_agreeing": winner_count,
            "distribution": dict(counts),
            "tree_results": tree_results,
        }


# ── 5. AdversarialReview ──────────────────────────────────────


@dataclass
class CritiqueResult:
    issues: list[dict]  # {"description": str, "severity": float}
    total_severity: float
    should_revise: bool


class AdversarialReview:
    """Self-critique loop: generate, critique, revise if severity exceeds threshold.

    Severity thresholds are placeholders — [RECALIBRATE-2.4B] after pretraining.
    The critic prompt asks the model to find flaws in its own reasoning.
    """

    # [RECALIBRATE-2.4B] These thresholds were set at 74.5M and need tuning
    SEVERITY_THRESHOLD = 0.5
    MAX_REVISIONS = 3

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        severity_threshold: float | None = None,
        max_revisions: int = 3,
        temperature: float = 0.3,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.severity_threshold = severity_threshold or self.SEVERITY_THRESHOLD
        self.max_revisions = max_revisions
        self.temperature = temperature

    def __call__(self, prompt: str) -> dict:
        response = self._generate(prompt)
        revisions = []

        for rev in range(self.max_revisions):
            critique = self._critique(prompt, response)
            revisions.append({
                "revision": rev,
                "response": response,
                "critique": critique,
            })

            if not critique.should_revise:
                break

            response = self._revise(prompt, response, critique)

        answer = extract_answer(response)
        return {
            "answer": answer,
            "final_response": response,
            "n_revisions": len(revisions),
            "revisions": revisions,
        }

    def _generate(self, prompt: str) -> str:
        device = (
            next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        out = self.model.generate(ids, max_new_tokens=512, temperature=self.temperature)
        return self.tokenizer.decode(out[0].tolist())

    def _critique(self, prompt: str, response: str) -> CritiqueResult:
        """Ask the model to critique its own response."""
        critic_prompt = (
            f"Review this solution for errors. For each issue found, "
            f"rate severity 0.0-1.0. Format: ISSUE: <description> SEVERITY: <float>\n\n"
            f"Question: {prompt}\n\nSolution: {response}\n\nCritique:"
        )
        critique_text = self._generate(critic_prompt)
        issues = self._parse_critique(critique_text)
        total = sum(i["severity"] for i in issues)
        return CritiqueResult(
            issues=issues,
            total_severity=total,
            should_revise=total > self.severity_threshold,
        )

    def _revise(self, prompt: str, response: str, critique: CritiqueResult) -> str:
        """Generate a revised response incorporating critique feedback."""
        issue_text = "\n".join(
            f"- {i['description']} (severity: {i['severity']:.1f})"
            for i in critique.issues
        )
        revise_prompt = (
            f"Revise your solution to fix these issues:\n{issue_text}\n\n"
            f"Original question: {prompt}\n"
            f"Original solution: {response}\n\n"
            f"Corrected solution:"
        )
        return self._generate(revise_prompt)

    def _parse_critique(self, text: str) -> list[dict]:
        """Extract structured issues from critique text."""
        issues = []
        for m in re.finditer(
            r"ISSUE:\s*(.+?)\s*SEVERITY:\s*(\d*\.?\d+)", text, re.IGNORECASE
        ):
            severity = float(m.group(2))
            severity = max(0.0, min(1.0, severity))
            issues.append({"description": m.group(1).strip(), "severity": severity})

        # Fallback: if model didn't follow format, assign default severity
        if not issues and len(text.strip()) > 20:
            issues.append({
                "description": text.strip()[:200],
                "severity": 0.3,  # [RECALIBRATE-2.4B]
            })
        return issues


# ── 6. DifficultyRouter ───────────────────────────────────────


class DifficultyRouter:
    """Entropy-based routing: estimate question difficulty, pick strategy.

    Difficulty tiers:
      - simple (entropy < low_threshold): direct generation
      - medium (low <= entropy < high_threshold): self-consistency voting
      - hard (entropy >= high_threshold): tree search or forest

    Thresholds are [RECALIBRATE-2.4B] — calibrate on held-out set after training.
    """

    # [RECALIBRATE-2.4B] Entropy thresholds from 74.5M experiments
    LOW_ENTROPY = 2.0
    HIGH_ENTROPY = 4.0

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        simple_strategy: SelfConsistencyVoter | None = None,
        medium_strategy: SelfConsistencyVoter | None = None,
        hard_strategy: ForestOfThought | TreeSearch | None = None,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.low_threshold = low_threshold or self.LOW_ENTROPY
        self.high_threshold = high_threshold or self.HIGH_ENTROPY

        # Default strategies if not provided
        self.simple = simple_strategy or SelfConsistencyVoter(
            model, tokenizer, n_samples=1, temperature=0.3
        )
        self.medium = medium_strategy or SelfConsistencyVoter(
            model, tokenizer, n_samples=16, temperature=0.7
        )
        self.hard = hard_strategy or TreeSearch(
            model, tokenizer, beam_width=4, max_depth=8
        )

    def __call__(self, prompt: str) -> dict:
        difficulty, entropy = self._estimate_difficulty(prompt)

        if difficulty == "simple":
            result = self.simple(prompt)
        elif difficulty == "medium":
            result = self.medium(prompt)
        else:
            result = self.hard(prompt)

        result["difficulty"] = difficulty
        result["entropy"] = entropy
        result["strategy"] = type(
            self.simple if difficulty == "simple"
            else self.medium if difficulty == "medium"
            else self.hard
        ).__name__
        return result

    @torch.no_grad()
    def _estimate_difficulty(self, prompt: str) -> tuple[str, float]:
        """Estimate question difficulty via model entropy on the prompt."""
        device = (
            next(iter(self.model.parameters())).device
            if hasattr(self.model, "parameters")
            else torch.device("cpu")
        )
        ids = torch.tensor(
            [self.tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        logits = self.model.forward(ids)
        entropy = compute_entropy(logits[0])

        if entropy < self.low_threshold:
            return "simple", entropy
        if entropy < self.high_threshold:
            return "medium", entropy
        return "hard", entropy


# ── 7. StrategyPipeline ───────────────────────────────────────


class StrategyPipeline:
    """Chains strategies based on difficulty routing.

    Pipeline:
    1. DifficultyRouter classifies the question
    2. For code tasks: ExecutionVerifier is always added
    3. For hard math: AdversarialReview wraps the output
    4. Returns the final answer with full trace
    """

    def __init__(
        self,
        model: GenerativeModel,
        tokenizer: Tokenizer,
        execution_verifier: ExecutionVerifier | None = None,
        adversarial_review: AdversarialReview | None = None,
        difficulty_router: DifficultyRouter | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = difficulty_router or DifficultyRouter(model, tokenizer)
        self.exec_verifier = execution_verifier or ExecutionVerifier(model, tokenizer)
        self.adversarial = adversarial_review or AdversarialReview(model, tokenizer)

    def __call__(self, prompt: str, task_type: str = "reasoning") -> dict:
        trace = {"task_type": task_type, "stages": []}

        # Stage 1: Route
        routed = self.router(prompt)
        trace["stages"].append({
            "stage": "routing",
            "difficulty": routed.get("difficulty"),
            "entropy": routed.get("entropy"),
            "strategy": routed.get("strategy"),
        })

        # Stage 2: Task-specific augmentation
        if task_type == "code":
            exec_result = self.exec_verifier(prompt)
            trace["stages"].append({"stage": "execution_verify", **exec_result})
            if exec_result["success"]:
                return {
                    "answer": exec_result["answer"],
                    "trace": trace,
                    "source": "execution_verifier",
                }

        # Stage 3: Adversarial review on hard questions
        if routed.get("difficulty") == "hard":
            reviewed = self.adversarial(prompt)
            trace["stages"].append({"stage": "adversarial_review", **reviewed})
            if reviewed["answer"] is not None:
                return {
                    "answer": reviewed["answer"],
                    "trace": trace,
                    "source": "adversarial_review",
                }

        return {
            "answer": routed.get("answer"),
            "trace": trace,
            "source": routed.get("strategy", "router"),
        }
