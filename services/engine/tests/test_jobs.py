"""Phase C1 tests: Celery job lifecycle, progress store, and the job endpoints.

No live Redis anywhere: jobs run in Celery eager mode against the in-memory
store, and the Redis store is exercised through a small in-process fake client.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app import jobs, main
from app.engine import scale

from test_scale import in_memory, make_csv


@pytest.fixture(autouse=True)
def eager_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("CSV_DOCTOR_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(main, "SSE_POLL_SECONDS", 0.01)
    jobs.configure("eager", jobs.MemoryJobStore())
    yield
    jobs.configure("eager", jobs.MemoryJobStore())


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    return make_csv(tmp_path_factory.mktemp("jobs") / "jobs.csv", n=1_500)


@pytest.fixture
def client():
    return TestClient(main.app)


def submit(client, path: Path, name="data.csv"):
    return client.post("/jobs", files={"file": (name, path.read_bytes(), "text/csv")})


def read_sse(client, url, headers=None) -> list[dict]:
    """Collect (event name, id, parsed data) for every SSE message until the stream closes."""
    out, current = [], {}
    with client.stream("GET", url, headers=headers or {}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        for line in resp.iter_lines():
            if line == "":
                if current:
                    out.append(current)
                    current = {}
            elif line.startswith(":"):
                continue
            else:
                key, _, value = line.partition(": ")
                current[key] = json.loads(value) if key == "data" else value
    return out


# -------------------------------------------------- Celery task + store (eager)

def test_job_lifecycle_submit_progress_result(csv_path, tmp_path):
    store = jobs.get_store()
    upload = tmp_path / "up.csv"
    upload.write_bytes(csv_path.read_bytes())
    job_id = "a" * 32
    jobs.submit_job(job_id, upload, "hash123", "test-version", chunk_rows=400)

    record = store.get(job_id)
    assert record["status"] == "done" and record["percent"] == 100.0
    assert record["error"] is None
    expected = in_memory(csv_path)
    assert record["result"]["profile"] == expected["profile"]
    assert record["result"]["health_score"] == expected["health_score"]
    assert record["result"]["file_hash"] == "hash123"
    assert record["result"]["engine_version"] == "test-version"

    events = store.events_after(job_id, 0)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[0]["status"] == "queued"
    running = [e for e in events if e["status"] == "running"]
    assert len(running) >= 4
    pcts = [e["percent"] for e in events]
    assert pcts == sorted(pcts)
    assert {e["stage"] for e in running} >= {"starting", "scanning chunks", "exact aggregates (DuckDB)"}
    assert all(e["percent"] < 100.0 for e in events[:-1])   # only the final event is at 100%
    assert events[-1]["status"] == "done" and events[-1]["summary"]["rows"] == expected["profile"]["rows"]
    assert store.events_after(job_id, events[-1]["seq"]) == []
    assert not upload.exists()   # the worker removes the upload when done


def test_failed_job_records_a_clean_error_and_removes_the_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"a,b\n")
    jobs.submit_job("b" * 32, bad, "h", "v")
    record = jobs.get_store().get("b" * 32)
    assert record["status"] == "failed" and record["result"] is None
    assert record["error"].startswith("empty_file")
    assert "Traceback" not in record["error"] and not bad.exists()


def test_unexpected_crash_is_reported_without_a_stack_trace(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(scale, "profile_file", boom)
    path = tmp_path / "x.csv"
    path.write_text("a\n1\n")
    jobs.submit_job("c" * 32, path, "h", "v")
    record = jobs.get_store().get("c" * 32)
    assert record["status"] == "failed" and "secret" not in record["error"]


def test_submit_rejects_non_hex_job_ids(tmp_path):
    with pytest.raises(ValueError):
        jobs.submit_job("../../etc/passwd", tmp_path / "x.csv", "h", "v")


def test_memory_store_is_bounded():
    store = jobs.MemoryJobStore(max_jobs=3)
    for i in range(5):
        store.create(f"{i:032x}")
    assert store.get(f"{0:032x}") is None and store.get(f"{4:032x}")["status"] == "queued"


# ------------------------------------------------------------ mode selection

def test_mode_resolution(monkeypatch):
    monkeypatch.setenv("CSV_DOCTOR_JOBS_MODE", "eager")
    assert jobs.configure() == "eager" and jobs.celery_app.conf.task_always_eager is True
    # auto + nothing listening on the port => in-process eager fallback, no exception
    monkeypatch.setenv("CSV_DOCTOR_JOBS_MODE", "auto")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    assert jobs.configure() == "eager"
    monkeypatch.setenv("CSV_DOCTOR_JOBS_MODE", "bogus")
    with pytest.raises(ValueError):
        jobs.configure()


def test_celery_urls_default_to_local_redis_and_read_the_environment(monkeypatch):
    for var in ("CELERY_BROKER_URL", "CELERY_RESULT_BACKEND", "REDIS_URL"):
        monkeypatch.delenv(var, raising=False)
    conf = jobs.build_celery().conf
    assert conf.broker_url == conf.result_backend == "redis://localhost:6379/0"
    monkeypatch.setenv("REDIS_URL", "redis://cache.internal:6380/2")
    conf = jobs.build_celery().conf
    assert conf.broker_url == conf.result_backend == "redis://cache.internal:6380/2"
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://broker.internal:6379/5")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://results.internal:6379/6")
    conf = jobs.build_celery().conf
    assert (conf.broker_url, conf.result_backend) == ("redis://broker.internal:6379/5",
                                                      "redis://results.internal:6379/6")


class FakeRedis:
    """Just the commands RedisJobStore uses."""

    def __init__(self):
        self.data, self.lists = {}, {}

    def pipeline(self):
        return self

    def set(self, key, value, ex=None):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.lists.pop(key, None)

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def expire(self, key, seconds):
        pass

    def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]

    def execute(self):
        return []


def test_redis_store_has_the_same_contract_as_the_memory_store():
    for store in (jobs.MemoryJobStore(), jobs.RedisJobStore(FakeRedis())):
        jid = "d" * 32
        assert store.get(jid) is None and store.events_after(jid, 0) == []
        store.create(jid)
        store.publish(jid, "running", 40.0, "scanning chunks")
        store.publish(jid, "done", 100.0, "complete", result={
            "profile": {"rows": 3, "cols": 1, "missing_pct": 0.0, "duplicate_rows": 0,
                        "duplicate_pct": 0.0, "columns": []},
            "health_score": 97, "score_reasons": []})
        assert store.get(jid)["status"] == "done" and store.get(jid)["result"]["health_score"] == 97
        events = store.events_after(jid, 0)
        assert [(e["seq"], e["status"]) for e in events] == [(1, "queued"), (2, "running"), (3, "done")]
        assert [e["seq"] for e in store.events_after(jid, 2)] == [3]
        assert events[-1]["summary"]["health_score"] == 97 and "result" not in events[-1]


# ------------------------------------------------------------------ the API

def test_post_jobs_then_status_returns_the_result(client, csv_path):
    resp = submit(client, csv_path)
    assert resp.status_code == 202
    body = resp.json()
    assert jobs.JOB_ID_RE.match(body["job_id"]) and body["status"] == "done"
    assert body["status_url"] == f"/jobs/{body['job_id']}"

    status = client.get(body["status_url"]).json()
    assert status["status"] == "done" and status["percent"] == 100.0
    expected = in_memory(csv_path)
    assert status["result"]["profile"] == expected["profile"]
    assert status["result"]["health_score"] == expected["health_score"]
    assert status["result"]["file_hash"] == __import__("hashlib").sha256(csv_path.read_bytes()).hexdigest()
    assert status["result"]["engine_version"] == main.ENGINE_VERSION


def test_sse_streams_progress_then_a_completion_event(client, csv_path):
    job = submit(client, csv_path).json()
    messages = read_sse(client, job["events_url"])

    names = [m["event"] for m in messages]
    assert names.count("progress") >= 1 and names[-1] == "complete"
    assert names.count("complete") == 1 and "failed" not in names
    ids = [int(m["id"]) for m in messages]
    assert ids == list(range(1, len(ids) + 1))
    progress = [m["data"] for m in messages if m["event"] == "progress"]
    assert all(p["status"] in ("queued", "running") for p in progress)
    assert any(p["stage"] == "scanning chunks" for p in progress)
    pcts = [m["data"]["percent"] for m in messages]
    assert pcts == sorted(pcts) and pcts[-1] == 100.0

    final = messages[-1]["data"]
    expected = in_memory(csv_path)
    assert final["status"] == "done"
    assert final["summary"]["health_score"] == expected["health_score"]
    assert final["summary"]["rows"] == expected["profile"]["rows"]
    assert "result" not in final        # the summary is small; the full result is GET /jobs/{id}


def test_sse_resumes_after_last_event_id(client, csv_path):
    job = submit(client, csv_path).json()
    everything = read_sse(client, job["events_url"])
    resumed = read_sse(client, job["events_url"], headers={"Last-Event-ID": everything[-2]["id"]})
    assert [m["event"] for m in resumed] == ["complete"]
    assert read_sse(client, job["events_url"], headers={"Last-Event-ID": "garbage"}) == everything


def test_sse_reports_a_failed_job(client):
    job = client.post("/jobs", files={"file": ("e.csv", b"a,b\n", "text/csv")}).json()
    assert job["status"] == "failed"
    messages = read_sse(client, job["events_url"])
    assert messages[-1]["event"] == "failed" and messages[-1]["data"]["error"].startswith("empty_file")
    assert client.get(job["status_url"]).json()["error"].startswith("empty_file")


def test_sse_follows_a_job_that_is_still_running(client):
    """A client that connects mid-job gets later events as they are published."""
    import threading
    store = jobs.get_store()
    jid = "e" * 32
    store.create(jid)

    def finish():
        store.publish(jid, "running", 50.0, "scanning chunks")
        store.publish(jid, "failed", 100.0, "failed", error="boom")

    timer = threading.Timer(0.15, finish)
    timer.start()
    try:
        messages = read_sse(client, f"/jobs/{jid}/events")
    finally:
        timer.join()
    assert [m["event"] for m in messages] == ["progress", "progress", "failed"]


def test_unknown_or_malformed_job_ids_are_404(client):
    for jid in ("f" * 32, "not-a-job", "..%2F..%2Fetc"):
        assert client.get(f"/jobs/{jid}").status_code == 404
        assert client.get(f"/jobs/{jid}/events").status_code == 404


def test_upload_limits_and_empty_upload(client, monkeypatch):
    assert client.post("/jobs", files={"file": ("e.csv", b"", "text/csv")}).status_code == 422
    monkeypatch.setattr(main, "MAX_JOB_BYTES", 10)
    resp = client.post("/jobs", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 10, "text/csv")})
    assert resp.status_code == 413 and "file_too_large" in resp.json()["detail"]
    assert list(jobs.upload_dir().iterdir()) == []      # nothing left behind


def test_broker_outage_is_a_503_and_leaves_no_file(client, csv_path, monkeypatch):
    def refuse(*_a, **_k):
        raise ConnectionError("broker down")
    monkeypatch.setattr(jobs.profile_file_task, "apply_async", refuse)
    resp = submit(client, csv_path)
    assert resp.status_code == 503
    assert list(jobs.upload_dir().iterdir()) == []


def test_store_outage_is_a_503_for_submit_and_status(client, csv_path):
    from redis.exceptions import ConnectionError as RedisConnectionError

    class DownStore(jobs.MemoryJobStore):
        def create(self, job_id):
            raise RedisConnectionError("redis is down")

        def get(self, job_id):
            raise RedisConnectionError("redis is down")

    jobs.configure("eager", DownStore())
    assert submit(client, csv_path).status_code == 503
    assert list(jobs.upload_dir().iterdir()) == []
    assert client.get("/jobs/" + "a" * 32).status_code == 503


def test_worker_has_the_task_registered_under_a_stable_name():
    assert "csv_doctor.profile_file" in jobs.celery_app.tasks
