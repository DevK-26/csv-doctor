# Ask-Your-Data (Phase B2)

`POST /ask` (multipart: `file` = the CSV, `question` = text) answers a question
about an uploaded CSV by **writing and running a pandas program**, never from
memory of the file, and shows a number only after it has been **verified**.

```
question ─▶ generator ─▶ {pandas code, plan} ─▶ sandbox ─▶ verify ─▶ answer
              ▲   one retry with the error fed back ◀── error ──┘      │
                                                       unverified / failed (honest)
```

## Response `status`

| status       | meaning                                                                 | `answer` |
|--------------|-------------------------------------------------------------------------|----------|
| `verified`   | program ran and matched an independent re-computation                    | set      |
| `unverified` | program ran but its result failed verification — **no number is shown**  | `null`   |
| `failed`     | no runnable program after one retry, or unanswerable (`reason` says why) | `null`   |

Every response includes the `program`, the `plan` it implements, the
`assumptions` made while reading the question, every `attempt`, and (when a
program ran) the verification `checks`. The UI should render `message` for
`unverified`/`failed` and never fall back to showing program output.

## Verification

The generator returns a **plan** (filters, metric, group-by, before/after
split) alongside the code. `verify.py` re-computes the plan in plain Python
(`math.fsum`, sorting, loops — no pandas aggregation, no code from the
program) and compares at 1e-9 relative tolerance, after dataset-derived sanity
checks (finite, counts within `[0, rows]`, mean/median/min/max inside the
column's range, group labels that exist). It proves *program ≡ plan*. It
cannot prove *plan ≡ what the user meant*, which is why the plan and the
assumptions are always returned next to the answer.

## Sandbox

Static policy check → isolated subprocess (empty env, empty temp cwd, rlimits
for CPU / memory / file size / open files, wall-clock kill of the process
group) → restricted builtins plus a CPython audit hook (no socket, ctypes,
subprocess, `os.*` mutation, no file open outside the Python library
directories, no writes). See `sandbox.py` for the exact guarantees and the
stated non-guarantee: this is not a kernel jail, so run the engine container
without network access in production.

## Upgrading to an LLM

The default `LocalRulesGenerator` is deterministic and needs no key. To use an
LLM, wrap any provider call in a function `complete(messages) -> str` and
override `get_generator` in `app/main.py`:

```python
from app.engine.ask import LLMCodeGenerator

def get_generator():
    return LLMCodeGenerator(my_provider_call, name="my-model")
```

The adapter sends only the masked schema context (column names, types, ranges,
low-cardinality value lists — never rows; values the facts masker would rewrite
are omitted) and requires a JSON reply `{plan, code, assumptions}` or
`{unanswerable}`. Note the categorical value lists *do* leave the engine for
the provider; that is what lets the model map "North" to `region == 'North'`.
Credentials belong in the provider function's environment, never in the repo.
The retry, sandbox and verification behaviour is identical for every generator;
an LLM that hallucinates a number is caught by verification
(`test_llm_that_hallucinates_a_number_is_caught_by_verification`).

## Local generator coverage

filter (`region is North`, `units over 50`, `excluding X`, `in March`,
`after 2025-03-15`) + group (`by`, `per`, `which`, `top N`) + aggregate
(count, sum, mean, median, min, max, distinct count) + before/after compare
(`did X change after March ...`). Not supported yet (answered with an explicit
`unsupported_question`): grouping by time period, grouping combined with a
before/after compare, joins, multi-column grouping.
