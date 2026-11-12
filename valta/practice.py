"""Self-practice: how Valta gets better without being asked.

The teacher (teacher_model) invents a small coding problem and tests for it. The problem is vetted before it is
trusted: the tests must fail against an empty stub, and the teacher must then solve the problem again, from
the text alone, in a way that passes them. Two independent writings that agree through a test is the check
that the problem is sound. The personal model (the 3B) then tries the same problem. When it already passes,
nothing is learned and the problem is dropped. When it fails, the teacher's verified solution is saved as a
training example for `valta train`, and that command's eval gate still decides whether the result replaces
the model you have.

This does not make the teacher smarter. The 3B can get as good as the teacher is at what tests can check,
and no better.

Problems never reuse an eval task: names, wording and near-matches are all filtered, so the eval stays a
fair test of what was learned.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from valta.config import Settings
from valta.evals import REPLY_SUFFIX, check_solution, contaminated, eval_prompts, extract_code, load_tasks
from valta.memory import Memory
from valta.ollama_client import OllamaError
from valta.prompt import build_system

SOURCE = "practice"
MIN_ASSERTS = 3
MAX_SOLUTION_CHARS = 2500  # longer answers will not fit in train_max_len with the root rules
MAX_CTX = 8192
SOLVE_TEMPERATURES = (0.2, 0.6)  # a second try, a little looser, when the first does not pass
SIMILARITY_LIMIT = 0.6
AVOID_SHOWN = 30

TOPICS = (
    "string cleanup and formatting",
    "counting and grouping with dictionaries",
    "sliding windows over a list",
    "recursion over nested lists or dicts",
    "sorting with custom keys and tie-breaking",
    "parsing a small text format (lines, key/value, CSV-like)",
    "number theory and divisors",
    "grids and matrices as lists of lists",
    "simulating a stack or a queue",
    "binary search and its off-by-one edge cases",
    "date and calendar arithmetic with the datetime module",
    "bit manipulation",
    "run-length encoding and other simple compression",
    "merging and splitting ranges",
    "graph traversal over a dict of neighbours",
    "dynamic programming on sequences",
    "generators and iterators",
    "tokenizing and evaluating simple expressions",
    "text wrapping, padding and alignment",
    "set operations and deduplication that keeps order",
    "prefix sums and running totals",
    "string comparison: anagrams, rotations, subsequences",
    "roman numerals, number bases and number-to-words",
    "scheduling and time slots",
    "validating input such as brackets, IDs or version strings",
    "priority queues with heapq",
    "fractions and exact decimal arithmetic",
    "caching and memoization",
)
LEVELS = ("easy", "medium", "hard")

# Model-written code is run on your machine. This keeps honest mistakes from touching files, processes or the
# network. It is not a defence against code written to evade it.
_RISKY = re.compile(
    r"\b(?:import|from)\s+(?:os|sys|subprocess|shutil|socket|ctypes|pathlib|requests|urllib|http|pickle|"
    r"importlib|builtins|multiprocessing|threading|signal|tempfile|glob|asyncio|platform)\b"
    r"|(?<![\w.])(?:open|exec|eval|__import__|input|breakpoint|exit|quit)\s*\("
    r"|\bos\.|\bsubprocess\.|\bsys\."
)
_NAME = re.compile(r"^[ \t]*NAME:[ \t]*`?([A-Za-z_][A-Za-z0-9_]*)`?[ \t]*$", re.M)
_PROMPT = re.compile(r"^[ \t]*PROMPT:[ \t]*(.+?)\s*\n[ \t]*TESTS:", re.M | re.S)
_TESTS = re.compile(r"TESTS:\s*```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S | re.I)
_WRITTEN = re.compile(r"\bWrite\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass
class Problem:
    name: str
    prompt: str
    tests: str

    @property
    def user_text(self) -> str:
        """Worded exactly like an eval prompt, so the model also learns the reply format."""
        return f"{self.prompt}\n\n{REPLY_SUFFIX}"


@dataclass
class Outcome:
    kind: str  # saved, known, rejected, unsolved
    name: str = ""
    note: str = ""


@dataclass
class Stats:
    tried: int = 0
    saved: int = 0
    known: int = 0
    rejected: int = 0
    unsolved: int = 0
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.tried += 1
        setattr(self, outcome.kind, getattr(self, outcome.kind) + 1)

    def summary(self) -> str:
        return (
            f"Tried {self.tried}: saved {self.saved}, already known {self.known}, "
            f"rejected {self.rejected}, unsolved {self.unsolved}."
        )


# ---------------------------------------------------------------------------------------------- checks


def parse_problem(reply: str) -> Problem | None:
    name, prompt, tests = _NAME.search(reply), _PROMPT.search(reply), _TESTS.search(reply)
    if not (name and prompt and tests):
        return None
    return Problem(name.group(1), " ".join(prompt.group(1).split()), tests.group(1).strip() + "\n")


def risky(text: str) -> bool:
    return bool(_RISKY.search(text))


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def too_close(text: str, prompts: list[str]) -> bool:
    """True when most of the wording matches an eval prompt, which an exact-substring check would miss."""
    mine = _words(text)
    for prompt in prompts:
        other = _words(prompt.replace(REPLY_SUFFIX, ""))
        if len(mine) >= 8 and len(other) >= 8 and len(mine & other) / len(mine | other) >= SIMILARITY_LIMIT:
            return True
    return False


def vet(problem: Problem, taken: set[str], blocked: list[str]) -> str | None:
    """Why a problem cannot be used, or None when it can. Cheap checks first; the stub run is last."""
    if problem.name in taken:
        return "name already used"
    if f"{problem.name}(" not in problem.prompt:
        return "the prompt does not show the function signature"
    if not 40 <= len(problem.prompt) <= 1200:
        return "the prompt is too short or too long"
    if risky(problem.tests):
        return "the tests touch files, processes or the network"
    if len(re.findall(r"^[ \t]*assert\b", problem.tests, re.M)) < MIN_ASSERTS:
        return f"fewer than {MIN_ASSERTS} asserts"
    try:
        compile(problem.tests, "tests", "exec")
    except SyntaxError:
        return "the tests do not parse"
    if contaminated(problem.prompt, blocked) or too_close(problem.prompt, blocked):
        return "too close to an eval task"
    # A test that a do-nothing function passes proves nothing.
    ok, _ = check_solution(f"def {problem.name}(*args, **kwargs):\n    return None\n", problem.tests)
    if ok:
        return "the tests pass without a solution"
    return None


def written_names(memory: Memory) -> set[str]:
    names: set[str] = set()
    for row in memory.all_examples():
        if row["source_id"] == SOURCE:
            names.update(_WRITTEN.findall(row["user_text"]))
    return names


# ---------------------------------------------------------------------------------------------- model calls


def write_problem(settings: Settings, client, topic: str, level: str, avoid: list[str]) -> Problem | None:
    system = build_system(settings, role="You write small, precise Python practice problems with tests.")
    user = (
        f"Write one new Python practice problem. Topic: {topic}. Difficulty: {level}.\n\n"
        "Rules:\n"
        "- The task is one pure function. No files, network, input(), randomness or clock. Use only the standard "
        "library (math, re, itertools, collections, functools, heapq, bisect, datetime, fractions, decimal).\n"
        "- The prompt must state every rule the tests check, in plain words: what to return for empty input, how "
        "ties are broken, the order of results, what counts as invalid. Someone who reads only the prompt must be "
        "able to pass every test.\n"
        '- The prompt is one short paragraph that starts with "Write NAME(args) that".\n'
        "- Give 5 to 8 plain assert lines, including edge cases that careless code gets wrong.\n"
        f"- Do not use these function names: {', '.join(avoid) or 'none'}.\n\n"
        "Reply in exactly this format and nothing else:\n"
        "NAME: <function_name>\n"
        "PROMPT: <the prompt>\n"
        "TESTS:\n"
        "```python\n"
        "assert ...\n"
        "```"
    )
    reply = client.chat(
        settings.heavy_host,
        settings.teacher_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        min(settings.teacher_num_ctx, MAX_CTX),
        temperature=0.9,
    )
    return parse_problem(reply)


def attempt(settings: Settings, client, host: str, model: str, ctx: int, problem: Problem, temperature: float):
    """One try at a problem. Returns (code or None, passed, note)."""
    system = build_system(settings, role="You are a careful Python programmer.")
    reply = client.chat(
        host,
        model,
        [{"role": "system", "content": system}, {"role": "user", "content": problem.user_text}],
        min(ctx, MAX_CTX),
        temperature=temperature,
    )
    code = extract_code(reply)
    if code is None:
        return None, False, "no code block"
    if risky(code):
        return code, False, "the solution touches files, processes or the network"
    ok, note = check_solution(code, problem.tests)
    return code, ok, note


# ---------------------------------------------------------------------------------------------- the loop


class Practice:
    """One practice session. `student_model=None` skips the check against the personal model."""

    def __init__(
        self,
        settings: Settings,
        memory: Memory,
        client,
        student_model: str | None = None,
        keep_known: bool = False,
        seed: int | None = None,
    ) -> None:
        self.settings = settings
        self.memory = memory
        self.client = client
        self.student_model = student_model
        self.keep_known = keep_known
        self.rng = random.Random(seed)
        self.blocked = eval_prompts(settings)
        self.eval_names = {task.id for task in load_tasks(settings)}
        self.written = written_names(memory)
        self.taken = set(self.eval_names) | set(self.written)
        self.stats = Stats()

    def once(self) -> Outcome:
        topic, level = self.rng.choice(TOPICS), self.rng.choice(LEVELS)
        recent = sorted(self.written)[-AVOID_SHOWN:]
        problem = write_problem(self.settings, self.client, topic, level, sorted(self.eval_names) + recent)
        if problem is None:
            return Outcome("rejected", note="the reply was not in the problem format")
        reason = vet(problem, self.taken, self.blocked)
        if reason:
            return Outcome("rejected", problem.name, reason)
        self.taken.add(problem.name)

        solution = None
        last_note = ""
        for temperature in SOLVE_TEMPERATURES:
            code, passed, last_note = attempt(
                self.settings,
                self.client,
                self.settings.heavy_host,
                self.settings.teacher_model,
                self.settings.teacher_num_ctx,
                problem,
                temperature,
            )
            if passed:
                solution = code
                break
        if solution is None:
            return Outcome("unsolved", problem.name, f"the teacher could not pass its own tests ({last_note})")
        answer = f"```python\n{solution.strip()}\n```"
        if len(answer) > MAX_SOLUTION_CHARS:
            return Outcome("rejected", problem.name, "the solution is too long to train on")

        note = "verified"
        if self.student_model:
            try:
                _, student_passed, _ = attempt(
                    self.settings,
                    self.client,
                    self.settings.self_host,
                    self.student_model,
                    self.settings.self_num_ctx,
                    problem,
                    0.0,
                )
            except OllamaError:
                self.student_model = None
                self.stats.notes.append("The personal model stopped answering, so later problems are saved unchecked.")
                note = "verified, personal model not checked"
            else:
                if student_passed and not self.keep_known:
                    return Outcome("known", problem.name, "the personal model already solves it")
                note = "verified, the personal model already solves it" if student_passed else "verified, the personal model fails it"
        if self.memory.example_exists(problem.user_text, answer):
            return Outcome("rejected", problem.name, "already saved")
        self.memory.add_example(problem.user_text, answer, source_id=SOURCE)
        self.written.add(problem.name)
        return Outcome("saved", problem.name, note)

    def run(self, count: int, on_progress: Callable[[int, Outcome], None] | None = None) -> Stats:
        for index in range(1, count + 1):
            try:
                outcome = self.once()
            except OllamaError as exc:
                self.stats.error = str(exc)
                break
            self.stats.add(outcome)
            if on_progress is not None:
                on_progress(index, outcome)
        return self.stats
