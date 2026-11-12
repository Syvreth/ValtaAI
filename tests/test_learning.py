import json
from pathlib import Path
from urllib.robotparser import RobotFileParser

from valta.config import load_settings, load_state
from valta.evals import EvalResult
from valta.external import has_code, import_dataset
from valta.memory import Memory
from valta.tools import ToolRunner
from valta.vectors import Vectors
from valta.web import in_scope, learn_site, normalize, scope_of


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


def _tools(settings, memory, project):
    return ToolRunner(settings, memory, Vectors(settings), lambda text: [0.1, 0.2, 0.3], project)


def _robots(text: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return parser


BODY = "Paths are built with os.path.join, which inserts the right separator for the platform. " * 6


def _site():
    def page(title, links, body=BODY, main=True):
        anchors = "".join(f'<a href="{link}">x</a>' for link in links)
        content = f"<main><h1>{title}</h1><p>{body}</p></main>" if main else f"<p>{body}</p>"
        return f"<html><head><title>{title}</title></head><body><nav>{anchors} MENU MENU</nav>{content}</body></html>"

    return {
        "https://docs.example.org/lib/": page(
            "Library",
            ["os.html#section", "os.html?highlight=x", "private/secret.html", "/blog/post.html",
             "https://other.example.org/lib/x.html", "data.zip", "path.html", "short.html"],
        ),
        "https://docs.example.org/lib/os.html": page("os module", ["/lib/"]),
        "https://docs.example.org/lib/path.html": page("os.path", [], main=False),
        "https://docs.example.org/lib/short.html": page("tiny", [], body="Too short."),
        "https://docs.example.org/lib/private/secret.html": page("secret", []),
    }


def test_scope_stays_on_the_site_and_under_the_path():
    scope = scope_of("https://docs.example.org/lib/os.html")
    assert scope == ("https", "docs.example.org", "/lib/")
    assert in_scope("https://docs.example.org/lib/path.html", scope)
    assert not in_scope("https://docs.example.org/blog/post.html", scope)
    assert not in_scope("https://other.example.org/lib/x.html", scope)
    assert not in_scope("http://docs.example.org/lib/x.html", scope)
    assert not in_scope("https://docs.example.org/lib/archive.zip", scope)
    assert normalize("https://docs.example.org/lib/os.html?highlight=x#part") == "https://docs.example.org/lib/os.html"


def test_learn_reads_politely_and_stores_main_text(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    tools = _tools(settings, memory, tmp_path)
    site = _site()
    fetched, waits = [], []

    def fetch(url):
        fetched.append(url)
        if url not in site:
            return 404, "text/html", ""
        return 200, "text/html; charset=utf-8", site[url]

    robots = _robots("User-agent: *\nDisallow: /lib/private/\nCrawl-delay: 2\n")
    stats = learn_site(settings, memory, tools, "https://docs.example.org/lib/", max_pages=10, delay=0.5,
                       fetch=fetch, robots=robots, sleep=waits.append)
    assert stats.pages == 3  # the index, os.html and path.html
    assert stats.skipped == {"robots.txt": 1, "too little text": 1}
    assert "https://docs.example.org/lib/private/secret.html" not in fetched
    assert not any("other.example.org" in url or "/blog/" in url or url.endswith(".zip") for url in fetched)
    assert fetched.count("https://docs.example.org/lib/os.html") == 1  # fragment and query collapse to one page
    assert waits and set(waits) == {2.0}  # the robots.txt Crawl-delay beats the shorter --delay
    stored = memory.recent(limit=20, kind="web")
    assert stored and all(record.url and record.url.startswith("https://docs.example.org/lib/") for record in stored)
    assert not any("MENU" in record.text for record in stored if record.title.startswith("os module"))
    again = learn_site(settings, memory, tools, "https://docs.example.org/lib/", max_pages=10, delay=0,
                       fetch=fetch, robots=robots, sleep=lambda seconds: None)
    assert again.pages == 0 and again.skipped.get("already learned") == 3


def test_learn_honours_a_page_limit_and_a_site_that_forbids_everything(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    tools = _tools(settings, memory, tmp_path)
    site = _site()

    def fetch(url):
        return (200, "text/html", site[url]) if url in site else (404, "text/html", "")

    one = learn_site(settings, memory, tools, "https://docs.example.org/lib/", max_pages=1, delay=0,
                     fetch=fetch, robots=_robots(""), sleep=lambda seconds: None)
    assert one.pages == 1
    blocked = RobotFileParser()
    blocked.disallow_all = True
    none = learn_site(settings, memory, tools, "https://docs.example.org/lib/os.html", max_pages=5, delay=0,
                      fetch=fetch, robots=blocked, sleep=lambda seconds: None)
    assert none.pages == 0 and none.skipped == {"robots.txt": 1}


def test_training_runs_resume_from_their_newest_checkpoint(tmp_path: Path):
    from valta.train import adapter_finished, latest_checkpoint, latest_run

    settings = _settings(tmp_path)
    assert latest_run(settings) is None
    dead = settings.adapters_dir / "20261112010000"  # stopped before its first checkpoint
    dead.mkdir(parents=True)
    finished = settings.adapters_dir / "20261111000000"
    finished.mkdir()
    (finished / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert latest_run(settings) == finished and adapter_finished(finished)
    unfinished = settings.adapters_dir / "20261112020000"
    for step in (25, 100, 75):
        (unfinished / f"checkpoint-{step}").mkdir(parents=True)
    (unfinished / "checkpoint-notes").mkdir()
    assert latest_checkpoint(unfinished).name == "checkpoint-100"
    assert latest_run(settings) == unfinished and not adapter_finished(unfinished)


class _Ollama:
    def __init__(self, digest, base_name="qwen2.5-coder:3b", base_digest="sha256:base"):
        self.digest = digest
        self.base_name = base_name
        self.base_digest = base_digest

    def tags(self, host):
        return [
            {"name": "valta:latest", "digest": self.digest},
            {"name": self.base_name, "digest": self.base_digest},
        ]


def _scored(monkeypatch, scores: dict[str, int]) -> list[str]:
    """Replace the eval with fixed scores out of 12. Returns the models it was asked to score, in order."""
    import valta.train as train

    runs = []

    def fake_eval(settings, client, host, model, on_progress=None):
        runs.append(model)
        return EvalResult(model, passed=scores[model], total=12)

    monkeypatch.setattr(train, "run_eval", fake_eval)
    return runs


def test_baseline_score_is_reused_until_the_model_changes(tmp_path: Path, monkeypatch):
    import valta.train as train

    settings = _settings(tmp_path)
    runs = _scored(monkeypatch, {"valta": 9})
    first = train.score_model(settings, _Ollama("sha256:aaa"), "valta")
    second = train.score_model(settings, _Ollama("sha256:aaa"), "valta")
    assert runs == ["valta"] and (second.passed, second.total) == (9, 12) and first.score == second.score
    train.score_model(settings, _Ollama("sha256:bbb"), "valta")  # new weights: score again
    assert runs == ["valta", "valta"]
    train.remember_baseline(settings, _Ollama("sha256:ccc"), "valta", EvalResult("valta-candidate", 11, 12))
    reused = train.score_model(settings, _Ollama("sha256:ccc"), "valta")
    assert runs == ["valta", "valta"] and reused.passed == 11
    assert load_state(settings)["baseline_cache"]["valta"]["fingerprint"].startswith("valta|sha256:ccc|")


def test_a_fine_tune_has_to_reach_the_base_model_too(tmp_path: Path, monkeypatch):
    """A worse fine-tune that got promoted must not become the bar the next one only has to tie."""
    import valta.train as train
    from valta.config import save_state
    from valta.evals import should_promote

    settings = _settings(tmp_path)
    save_state(settings, {"personal_model": "valta"})
    scores = {"valta": 9, settings.self_model: 10}
    runs = _scored(monkeypatch, scores)
    client = _Ollama("sha256:aaa", base_name=settings.self_model)

    bar = train.score_baseline(settings, client)
    assert (bar.model, bar.passed) == (settings.self_model, 10)
    assert not should_promote(bar, EvalResult("valta-candidate", 9, 12), 0.0)[0]  # ties valta, loses to the base
    assert should_promote(bar, EvalResult("valta-candidate", 10, 12), 0.0)[0]

    train.score_baseline(settings, client)  # nothing changed: both scores come from the cache
    assert runs == ["valta", settings.self_model]

    scores["valta"] = 11  # new valta weights that beat the base: valta is the bar again
    bar = train.score_baseline(settings, _Ollama("sha256:bbb", base_name=settings.self_model))
    assert (bar.model, bar.passed) == ("valta", 11)
    assert runs == ["valta", settings.self_model, "valta"]


def test_the_base_model_is_scored_once_when_it_is_the_personal_model(tmp_path: Path, monkeypatch):
    import valta.train as train
    from valta.config import save_state

    settings = _settings(tmp_path)
    save_state(settings, {"personal_model": settings.self_model})
    runs = _scored(monkeypatch, {settings.self_model: 10})
    bar = train.score_baseline(settings, _Ollama("sha256:aaa", base_name=settings.self_model))
    assert runs == [settings.self_model] and bar.passed == 10


def test_a_score_saved_by_an_older_version_is_still_reused(tmp_path: Path, monkeypatch):
    import valta.train as train
    from valta.config import save_state

    settings = _settings(tmp_path)
    client = _Ollama("sha256:aaa")
    old = {"fingerprint": train.baseline_fingerprint(settings, client, "valta"), "passed": 9, "total": 12, "time": "t"}
    save_state(settings, {"personal_model": "valta", "baseline_cache": old})
    runs = _scored(monkeypatch, {"valta": 0})
    assert train.score_model(settings, client, "valta").passed == 9 and runs == []


def test_max_examples_trims_imported_pairs_first(tmp_path: Path):
    from valta.train import build_pairs

    settings = _settings(tmp_path)
    memory = Memory(settings)
    for index in range(3):
        memory.add_example(f"my question {index}", f"my answer {index}")
    for index in range(10):
        memory.add_external("ds", str(index), [{"role": "user", "content": f"q{index}"},
                                               {"role": "assistant", "content": f"a{index}"}])
    path, ids = build_pairs(settings, memory, client=None, max_examples=5)
    users = [json.loads(line)["messages"][1]["content"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(users) == 5 and sum(user.startswith("my question") for user in users) == 3
    path, _ = build_pairs(settings, memory, client=None, max_examples=2)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3  # your own examples are never cut


def test_import_can_require_long_answers_with_code(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)

    def record(rid, answer):
        return {"id": rid, "langdetect": "en", "label": {"ability_en": ["code"]},
                "conversations": [{"from": "human", "value": "Write it."}, {"from": "gpt", "value": answer}]}

    records = [
        record("1", "Returns the byte size."),
        record("2", "Here is a long explanation without any code at all. " * 6),
        record("3", "Here you go:\n```python\ndef size(x):\n    return len(x)\n```\nIt returns the length. " * 2),
    ]
    stats = import_dataset(settings, memory, "ds", records=records, min_answer=100, code_only=True)
    assert stats.kept == 1 and stats.skipped == {"short answer": 1, "no code": 1}
    assert has_code("def f():\n    pass") and has_code("```\nx\n```") and not has_code("Define the function.")


def test_training_questions_come_from_books_and_web_pages(tmp_path: Path):
    from valta.train import document_passages

    settings = _settings(tmp_path)
    memory = Memory(settings)
    memory.add_record("A book passage.", kind="book", title="Book #1")
    memory.add_record("A web passage.", kind="web", title="Docs #1", url="https://docs.example.org/x")
    memory.add_record("A research note.", kind="research", title="Note")
    texts = [record.text for record in document_passages(memory, 5)]
    assert sorted(texts) == ["A book passage.", "A web passage."]
    assert len(document_passages(memory, 1)) == 1


def test_lean_prepare_keeps_embeddings_16_bit_and_freezes_the_base():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from valta.train import prepare_lean

    config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).to(torch.float16)
    upcast = prepare_lean(model)
    assert model.get_input_embeddings().weight.dtype == torch.float16  # the big table stays 16-bit
    norms = [p for name, p in model.named_parameters() if "norm" in name]
    assert norms and all(p.dtype == torch.float32 for p in norms)
    assert upcast == sum(p.numel() for p in model.parameters() if p.ndim == 1)
    assert not any(p.requires_grad for p in model.parameters())
    assert model.is_gradient_checkpointing


def test_benchmark_report_leaves_out_warm_up_and_flags_a_full_card(tmp_path: Path):
    from valta.train import _StepTimer, benchmark_report

    settings = _settings(tmp_path)
    gib = 1024**3
    report = benchmark_report([90.0, 40.0, 42.0, 41.0], int(3.1 * gib), int(3.5 * gib), 6 * gib, settings)
    assert "median 41.0 s/step" in report and "first step 90.0 s" in report and "spilling" not in report
    full = benchmark_report([60.0, 61.0], int(5.6 * gib), int(5.9 * gib), 6 * gib, settings)
    assert "spilling" in full
    assert benchmark_report([], 0, 0, gib, settings) == "No steps finished."

    class Base:
        pass

    timer = _StepTimer(Base)
    timer.on_step_begin(None, None, None)
    timer.on_step_end(None, None, None)
    assert len(timer.durations) == 1 and timer.durations[0] >= 0
