# CS4603 PA4 — Document Analyst (Student Submission)

> This is your **submission file**. `README.md` is the assignment spec — this document is where you write up your work.
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment spec.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

My `.env` targets the Unity Catalog **`27100082_pa4`** (schema `default`) — note this workspace
has **no `main` catalog** (the spec's `main.default` is a placeholder), so I created a
`27100082_pa4` catalog via the Catalog Explorer UI (API creation is blocked by the metastore's
Default-Storage policy). Key values:

```
DATABRICKS_HOST=https://dbc-70e1cecd-4a2a.cloud.databricks.com
DATABRICKS_MODEL=databricks-meta-llama-3-3-70b-instruct
EMBEDDINGS_ENDPOINT=databricks-gte-large-en
UC_CATALOG=27100082_pa4
UC_SCHEMA=default
VECTOR_SEARCH_ENDPOINT=27100082-vs-endpoint
VECTOR_SEARCH_INDEX=27100082_pa4.default.27100082_analyst_index
SOURCE_TABLE=27100082_pa4.default.27100082_analyst_chunks
SERVING_ENDPOINT_NAME=27100082-document-analyst
SECRET_SCOPE=cs4603-deploy
```

`config.py` loads `.env` and validates the required vars; the LLM and Vector Search index are
managed Databricks services reachable with `DATABRICKS_HOST`/`DATABRICKS_TOKEN`, so the same code
runs locally and inside the serving container.

## Running locally

**1. Verify the MCP server (Task 0.2)** — the given math/finance tool server starts and exposes
its 5 tools:

```bash
uv run python tools/mcp_server.py     # starts (stdio, waits for input)
```

**2. Ingest the corpus (Task 0.3)** — the parse/chunk step uses Spark + `ai_parse_document` /
`ai_prep_search`, which only exist on Databricks, so this runs **once in a Databricks notebook**.
I created a UC Volume, uploaded the PDF, then ran:

```python
from rag.ingest import ingest
ingest(spark, "/Volumes/27100082_pa4/default/pa4/annual_report.pdf")
```

`rag/ingest.py` parses the PDF, chunks it into the Delta table
`27100082_pa4.default.27100082_analyst_chunks` (Change-Data-Feed enabled), creates a **STANDARD**
Vector Search endpoint and a **TRIGGERED Delta-Sync index** with managed embeddings
(`primary_key="chunk_id"`, `embedding_source_column="chunk_to_retrieve"`), then waits for the
index to reach `READY` and runs a similarity smoke query. After that the index is reachable from my
laptop too — `rag/store.py::get_retriever()` returns the identical retriever locally and in the
container.

**3. Build and run the graph** (`pa4.ipynb`, or any local script):

```python
from agent.graph import build_graph
graph = build_graph()          # config.py LLM + rag/store.py retriever + MCP tools
result = graph.invoke({"messages": [{"role": "user",
          "content": "What was the net revenue in 2023?"}]})
print(result["messages"][-1].content)
```

**4. Offline smoke test** (no Databricks, mocked LLM — the Bonus-A CI target):

```bash
uv run --extra dev python -m pytest tests/test_smoke.py -q   # 2 passed
```

**Test queries I ran** (all correct against the report; see `pa4.ipynb` for full traces):

| Query | Answer produced |
|-------|-----------------|
| "What was the net income in 2023?" | ¥1,107 billion [source: annual_report.pdf, p.3] |
| "What is 15% of 2.4 billion?" | 0.36 billion |
| "What was 2023 revenue, and a 10% increase?" | ¥16.91T → ¥18.601T (16.91 × 1.10) |

## Deployment

The model is packaged with MLflow **models-from-code** (`deployment/agent_model.py`), registered in
Unity Catalog, and served on a Databricks Model Serving endpoint. Everything is driven by
`deployment/deploy.py`:

```bash
python deployment/deploy.py --skip-secrets     # log + register + create/update endpoint
```

- **Log + register (2.2):** `mlflow.langchain.log_model(lc_model="deployment/agent_model.py",
  code_paths=["agent","rag","tools","config.py"], pip_requirements=[…pinned…])` then
  `mlflow.register_model(...)` into `27100082_pa4.default.document_analyst`. Shipping `code_paths` is
  essential — without it the container fails at startup with `ModuleNotFoundError: No module named
  'agent'`.
- **Endpoint (2.3):** `WorkspaceClient().serving_endpoints` with `workload_size="Small"`,
  `scale_to_zero_enabled=True`. Credentials are injected as **secret references**
  (`{{secrets/cs4603-deploy/…}}`); the Vector-Search vars are passed as plaintext env vars so the
  container's retriever can reach the index.
- **Endpoint name / URL:** `27100082-document-analyst` →
  `…/serving-endpoints/27100082-document-analyst/invocations` (serving **version 6**).
- **Result (2.4):** HTTP 200, non-empty cited answers, ~5 s warm latency; local and deployed answers
  are **identical** (temperature 0). The endpoint returns **Path A** — raw LangGraph state as a
  one-element batch list — so responses are parsed as `data[0]["messages"][-1]["content"]` via a
  direct `requests.post` (the OpenAI `.choices` shape does not apply).

**Debugging story (the reproducibility that matters most).** Logging on **Windows + Python 3.14**
caused a chain of container failures that all trace back to the cross-OS / cross-version mismatch:
1. Python **3.14** baked into the model — Databricks Serving can't build a 3.14 image (no wheels →
   pip source-builds and backtracks through hundreds of old `regex` releases → 40-min build timeout,
   retry loop). **Fix:** log from a Python **3.12** environment (added a `.python-version` file so
   MLflow's uv detection bakes 3.12).
2. **Unpinned** `pip_requirements` amplified the backtracking. **Fix:** pin exact versions, including
   the transitive `regex`/`tiktoken`.
3. Model-code path baked as a **Windows absolute path with backslashes**; the Linux loader's
   `os.path.basename` doesn't split on `\`, giving `/model/D:\…\agent_model.py` → FileNotFoundError.
   **Fix:** `deploy.py` forces forward slashes in the recorded `model_code_path`.
4. The MCP stdio subprocess launched with a bare `"python"` (wrong interpreter). **Fix:** launch with
   `sys.executable`.

Net lesson: deploying from a Databricks (Linux) notebook would have avoided the Python-version and
path-separator problems entirely; logging from a matched Python 3.12 environment is the fix when
deploying from a local Windows machine.

## Design decisions

- **Plan → Supervise → Specialise → Synthesise.** The graph decomposes a query into 2–5 atomic
  steps (planner), routes each step to a specialist (supervisor), and only synthesises at the end.
  Separating retrieval from computation lets each specialist have a focused prompt and be tuned
  independently, and keeps an auditable per-step trace (which the citations and the analysis rubric
  rely on).
- **Typed `AnalystState` with `messages` as the entry/exit channel.** `messages` uses the
  `add_messages` reducer (append, not overwrite); the other fields (`plan`, `current_step_index`,
  `step_results`, `next_agent`, `final_answer`) are internal scratch space. The synthesizer writes
  the answer to **both** `final_answer` and `messages` (as an `AIMessage`) — required so the
  messages-in/messages-out serving contract returns a non-empty answer.
- **Supervisor routing with a deterministic backstop.** The supervisor classifies each step with the
  LLM, but falls back to keyword hints (e.g. "CAGR", "%", "growth" → math) if the LLM answer is
  ambiguous, and the conditional edge defaults to synthesis rather than looping forever.
- **Deterministic math via a bundled MCP server.** Calculations go to real Python tools (no LLM
  arithmetic/hallucination). Tools are loaded once at graph-build time and invoked synchronously; the
  async/subprocess handling is the most fragile part of deployment, so `_run_async` uses a
  subprocess-capable event loop and runs on a worker thread when a loop is already active.
- **All-Databricks retrieval.** Using a managed Vector Search index (not a local vector store) means
  the identical retriever runs locally and in the container with no separate embedding path — the
  main reason the deployed model needed no retrieval code changes.
- **Client SDK resilience.** `DocumentAnalystClient` reads credentials from the environment, retries
  429/503 with exponential backoff, raises a clear `TimeoutError` with elapsed time, exposes a
  `READY`-gated `health_check()`, wraps failures in `AnalystClientError` (status + request id), and
  degrades `ask_streaming()` to a single full-answer chunk because the models-from-code endpoint does
  not emit token deltas.

---

## Analysis Questions

> Answer in your own words. Each question is copied from the assignment so you don't have to flip back.

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   - Dependencies are handled by **ordering + shared state**, not by passing values between steps explicitly. The planner is prompted to emit steps in execution order (retrieval before the computation that uses it), and steps are executed strictly sequentially via `current_step_index`. Crucially, every specialist node reads the *whole* `AnalystState`, and each finished step appends its output to `step_results`. So when a later computation step runs (e.g. "apply 8% growth"), the MCP node can see the revenue figure that the earlier retrieval step already wrote into `step_results` and use it as the operand. The dependency is resolved at *execution* time by reading accumulated results, rather than the planner trying to hard-code the value it doesn't yet know.
   - Limitation: because the plan is a list of static strings written up-front, a step's text can't contain a value that only exists after an earlier step runs. Our design tolerates this because the executing node is given the running `step_results` as context and extracts the needed number itself. The place this breaks is a *branching* dependency (step 3's very existence depends on step 1's result) — a purely sequential, pre-computed plan can't express that; it would need replanning (Q2).

2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   - For this corpus (a single, well-structured annual report answering fairly direct analytical questions) replanning after every step would mostly **hurt**: it adds an extra LLM call per step (latency + cost) and introduces a chance for the model to *revise a correct plan into a worse one*, all for queries whose decomposition is stable and knowable up front. Example: "revenue in FY2023, then +8% for 3 years" is fully plannable before any execution — replanning buys nothing but two extra LLM round-trips.
   - Replanning would **help** for open-ended or conditional queries where a later step depends on what an earlier step *found*. Example: "Which segment had the lowest operating margin, and what would its profit be if that margin rose to the group average?" — you can't write the growth step until retrieval reveals *which* segment is lowest. There, a replan after step 1 lets the plan adapt to the discovered fact. So the right call is conditional replanning (replan only when a step result invalidates the remaining plan), not replanning on every step.

### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
   - The failure mode is a *capability mismatch*: the specialist receives a step it can't do well. If a computation step ("apply 8% growth") is sent to `rag_agent`, the vector store returns semantically-similar prose instead of doing arithmetic, producing a wrong or fabricated number. If a retrieval step is sent to `mcp_tools`, the math tools have no figure to operate on and return an error or nonsense. Either way the bad result lands in `step_results` and the synthesizer faithfully reports it, so a misroute becomes a *silent* wrong answer.
   - **Detection:** each specialist can self-report confidence — the RAG node already returns "not found in documents" on empty retrieval, and the MCP path fails loudly if no numeric operand is available. So a mismatched route tends to surface as an empty/failed step result. To catch it explicitly I'd add a lightweight validity check: a computation step whose accumulated context contains no numbers, or a retrieval step that returns zero chunks, is flagged as a probable misroute.
   - **Recovery:** on a flagged step, route it once to the *other* specialist (a single fallback re-route) before giving up; the shared `step_results` means nothing is lost by retrying. A cheap hardening I already use is the deterministic keyword backstop in `_classify_step` — if the LLM's answer is ambiguous, obvious math verbs ("calculate", "CAGR", "%") force `mcp_tools`, which removes the most common misroute without any extra LLM cost. In a stricter system I'd cap re-routes (e.g. one retry) to avoid loops and log misroutes for prompt tuning.

2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   - A single ReAct agent interleaves reasoning and tool calls in one loop with one big prompt: simpler to build, fewer LLM hops, and it can improvise. Its weaknesses are exactly what this assignment stresses: with retrieval *and* math tools in one context the model is more prone to tool-selection mistakes, to hallucinating arithmetic instead of calling the calculator, and to muddling multi-step plans; its behaviour is also harder to audit because planning and execution are entangled.
   - The supervisor (plan → route-per-step → specialist) is worth the extra complexity when (a) the task reliably mixes *distinct* capabilities that benefit from separate, focused prompts (document lookup vs. deterministic calculation — precisely our case), (b) you need an **auditable trace** of what was decided and why (the plan and per-step routing are explicit, which the rubric and the synthesizer's citations rely on), and (c) you want to tune or swap one specialist (e.g. change the retriever's `k`, or move MCP to a remote server in Bonus C) without touching the others. For a simple single-tool or purely conversational task, that structure is over-engineering and a ReAct agent is the better trade.

### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   - Retrieving per-step generally *improves* precision. A decomposed step like "Find Meridian's net revenue for fiscal year 2023" is a single, focused information need, so its embedding points at one region of the document and the top-k chunks are tightly on-target — in testing it returned the exact ¥16.91 trillion figure. The original combined question ("...and what would it be after 3 years of 8% growth?") mixes a retrieval intent with a computation intent; its embedding is a blurry average of both, which can pull in growth/forecast chunks that dilute the top-k and push the actual revenue figure down the ranking. Focused sub-queries also let each step use a small `k` without missing the answer.
   - The trade-off is *lost context*: a step retrieved in isolation can be ambiguous where the full question would have disambiguated it (e.g. a bare "find the operating margin" step loses which segment or year the user meant). We mitigate this by having the planner write self-contained steps (it includes the entity/year in the step text), but the general risk is that over-decomposition strips context the retriever needs. Net: per-step retrieval trades a little context for a lot of precision, which is the right trade for this well-structured report.

2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   - Rewrite/expand the query before embedding it. Concretely: (a) **carry context from the original question and prior `step_results`** — e.g. inject the entity ("Meridian"), fiscal year ("FY2023"), and any metric named earlier, turning "find relevant financial data" into "Meridian FY2023 net revenue, operating profit, and net income"; (b) **query expansion** — have the LLM propose 2-3 concrete sub-queries (a HyDE-style hypothetical answer, or multi-query retrieval) and union the retrieved chunks; (c) **constrain with metadata** where possible (restrict to the source doc / relevant pages). I'd also raise `k` for vague steps and let the extraction prompt discard irrelevant chunks. The cheapest effective version here is the context-injection rewrite, since the planner already knows the entity and year and can bake them into the step text so the vague step never reaches the vector store as-is.

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   - `models-from-code` serialises the model by saving the *source file* and re-executing it in a fresh serving container, not by pickling live Python objects. So at load time the container re-runs `agent_model.py` from scratch with only what was shipped: the packages listed in `code_paths` (`agent`, `rag`, `tools`, `config.py`), the deps in `pip_requirements`, and the `environment_vars`. Anything the file *assumes already exists in memory or on the local machine* is gone. If the file referenced external state — a pgvector DB on `localhost`, an open connection object, a file at a laptop path, an in-process variable — the re-execution fails: the container has no `localhost` database and no laptop filesystem, so import raises (or, worse, silently misbehaves at inference). That is exactly why our file rebuilds everything from importable code + env-configured *managed* services (the LLM endpoint and the Vector Search index), and validates its env vars at import so the failure is loud and named rather than a cryptic `DEPLOYMENT_FAILED`.
2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - **Freshness:** an external index wins — you can re-ingest/re-sync the corpus (or point at a new index) without rebuilding or redeploying the model. A baked-in corpus is frozen at build time; updating it means logging and deploying a new model version.
   - **Cold-start size / build:** external wins — the artifact stays tiny (just code), so container builds and cold starts are faster and cheaper. Baking the corpus (documents + a vector store + possibly the embedding model) bloats the image, slowing every build and cold start, and duplicates the corpus into every model version.
   - **Latency:** baked-in wins on the steady state — an in-process lookup avoids a network round-trip to the index on every query, so per-request latency is lower and more predictable. The managed index adds a network hop (and, on scale-from-zero, the index may itself need to warm).
   - **Failure modes:** they differ. External introduces *runtime* dependencies — the endpoint now fails if the VS index is down, not `READY`, mis-permissioned, or its env vars are missing (which is why `rag/store.py` and `agent_model.py` validate those up front). Baked-in removes that network dependency (more self-contained, works offline) but shifts failure to *build/deploy* time and to staleness — a bug or gap in the corpus is shipped and can only be fixed by redeploying.
   - **Net for PA4:** the managed-index approach is the right default here — the corpus is shared, may change, and keeping the artifact tiny means the identical retriever code runs locally and in the container with no separate embedding path. You would prefer baking the corpus in only when it is small, static, latency-critical, and must run without depending on an external service.

### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   - Two different authentications are involved. The platform authenticates the *incoming request* to the endpoint (is the caller allowed to invoke this endpoint), and that is handled for you. But our model, once running, makes its own *outbound* calls to other Databricks services at inference time — the LLM serving endpoint (`ChatOpenAI` with `base_url=$DATABRICKS_HOST/serving-endpoints`) and the Vector Search index. Those outbound calls need their own credentials, and the serving container has no `.env` and no ambient user identity. So `DATABRICKS_TOKEN`/`DATABRICKS_HOST` are injected as env vars (via the secret scope) precisely so the model can authenticate *as* a principal when it calls the LLM and the retriever. Without them, the endpoint itself starts, but every request fails with 401 when the graph tries to reach the LLM or the index. (In production you would prefer a service principal / on-behalf-of token over a personal PAT, but the mechanism is the same: the running model needs its own outbound credential.)
2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - Databricks does a **rolling, zero-downtime update**. It provisions the new version's served entity alongside the currently-serving one and only shifts traffic once the new version passes its health checks and is `READY`; the old version keeps serving until then, and is torn down afterward. In-flight requests that were already accepted by the old version continue to be handled by it, so they are not dropped mid-flight; new requests are routed to the new version once it is live. The endpoint's `state` shows `config_update = IN_PROGRESS/UPDATING` during the swap and returns to `NOT_UPDATING` when done — which is exactly what `update_config_and_wait` blocks on. The practical caveats: the switch is at request boundaries (a single request isn't migrated between versions), and if you did a scale-to-zero cold start the very first post-swap request can be slow while the new container warms.

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - The 429/503 responses we retry on are usually caused by the endpoint being *overloaded or cold-starting* (scaling from zero). A fixed short interval means every client hammers the endpoint again at the same cadence, adding load exactly when it is already struggling — and if many clients retry in lockstep, they synchronize into repeated traffic spikes that keep the endpoint saturated. Exponential backoff (1s, 2s, 4s, … in our client) spaces retries out geometrically, so the endpoint gets progressively more breathing room to scale up or drain its queue, and the total retry traffic decays instead of staying constant. It also matches the timescale of the failure: a scale-from-zero cold start takes tens of seconds, so waiting longer between later attempts is more likely to hit a ready endpoint than retrying every second. (In production you'd add jitter to the backoff to prevent many clients from retrying at the exact same moments — the "thundering herd" problem.)

2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - A high `max_retries` turns a transient outage into a self-inflicted, amplified one. If the endpoint is down or throttling and every one of N concurrent users retries, say, 10 times, you multiply the offered load ~10x precisely when the service is unhealthy — a retry storm that can prevent it from ever recovering (and, with scale-to-zero autoscaling, can even drive it to scale *up* and burn cost chasing failing requests). It also ties up client resources: each request holds a worker/thread/connection far longer (retries × backoff can be minutes), so upstream callers queue up, latency explodes, and you can exhaust connection pools or thread pools — cascading the failure into the caller. The safer posture is a small `max_retries` with exponential backoff **and jitter**, plus a circuit breaker that stops retrying once a failure rate threshold is crossed, so a struggling endpoint is given room to recover instead of being pinned down.

3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - Choose `ask_streaming()` when the answer is long and *perceived latency* matters more than getting the complete result atomically — i.e. any interactive, user-facing surface. Concrete example: a chat UI in a financial-analysis web app. With `ask()`, the analyst spends several seconds retrieving and computing, and the user stares at a spinner until the entire paragraph arrives at once. With `ask_streaming()`, the first words ("Meridian's FY2023 net revenue was ¥16.91 trillion…") appear as soon as they are generated and the rest fills in progressively, so the user sees immediate progress, can start reading, and can abort early if it's the wrong direction — the interaction *feels* far faster even though total time is similar. You'd stick with `ask()` for non-interactive/programmatic use (batch jobs, another service consuming the answer, storing to a database) where nothing reads partial output and a single clean value is simpler to handle. (Note: our deployed models-from-code endpoint does not actually emit token deltas, so `ask_streaming()` currently falls back to yielding the full answer once — the *interface* is streaming-ready for a `ChatAgent`-style endpoint that does stream.)

### Bonus A — CI/CD (attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - `main` is the single, protected, reviewed source of truth — merging to it is the deliberate "this is ready" signal. Feature branches are experimental and frequently broken or half-finished; deploying from them would push untested work to the *one* live endpoint that real clients hit. Because all branches would target the same endpoint, concurrent feature-branch deploys would also race and overwrite each other's model versions, making it impossible to know what is actually serving. Restricting deploy to `main` (and excluding `pull_request` events) means PRs still get lint+test feedback for reviewers, but only a reviewed merge causes a real deployment — my workflow enforces this with `if: github.ref == 'refs/heads/main' && github.event_name != 'pull_request'` plus `needs: lint-and-test`.
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - Add an **evaluation gate** between `test` and `deploy`. After logging the candidate model but before updating the endpoint, run it against a fixed, held-out eval set (representative question→expected-answer pairs) and compute a quality metric — e.g. answer correctness / faithfulness / citation accuracy via `mlflow.evaluate`. Fetch the currently-serving version's logged metric and compare: if the candidate scores below the production version (or below an absolute threshold), fail the job so the deploy step never runs. This turns "did the code compile" (what lint+test checks) into "is the model actually at least as good," preventing silent quality regressions. Practical refinements: require a margin (candidate must beat prod by ≥ ε to account for eval noise), keep the eval set version-controlled and small enough to run in CI, and log the eval report as a build artifact so a human can inspect a borderline result.

### Bonus B — `databricks-agents` SDK (attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - **`agents.deploy()` (gain: speed/convenience; lose: fine-grained control).** One call provisions the serving endpoint, wires authentication automatically (no secret scope to create/manage), and additionally stands up a **Review App** and the evaluation/feedback plumbing. You get the production niceties (AI Playground integration, human-feedback capture, tracing) for free. What you give up is explicit control over the endpoint's construction — the SDK decides the endpoint name and much of the config, so custom `environment_vars`, precise `served_entities` layout, traffic splitting, or bespoke secret wiring are less directly in your hands, and there's more "magic" to debug if something goes wrong.
   - **Manual MLflow + `WorkspaceClient` (Part 2) (gain: control/transparency; lose: convenience).** You call `log_model` → `register_model` → `serving_endpoints.create` yourself, so you control every knob (endpoint name, `workload_size`, `scale_to_zero`, exact secret references, env vars) and you *see* every step — which is exactly why we could diagnose the Python-version / path / dependency failures. The cost is more boilerplate and you manage the secret scope and endpoint lifecycle by hand, with no Review App unless you build one.
   - **When to use which:** learn/debug and retain control with the manual path (Part 2); ship an agent that needs human feedback and evaluation quickly with `agents.deploy()`.
2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - Reviewers use the Review App to ask questions and rate each answer (thumbs up/down + a correction/comment), and that feedback is logged to the MLflow experiment alongside the request, the retrieved context, and the trace. Concrete loop: (1) **Collect** — accumulate rated interactions; (2) **Triage** — filter for thumbs-down and read the trace to attribute the failure (bad retrieval? a misroute by the supervisor? wrong arithmetic? weak synthesis?); (3) **Turn failures into an eval set** — promote corrected examples into a version-controlled question→expected-answer dataset; (4) **Fix the attributed cause** — e.g. tune the retriever `k`/query rewriting, sharpen the supervisor/planner prompts, or add a tool — and re-run `mlflow.evaluate` on the eval set to confirm the metric improves and nothing regresses; (5) **Gate + ship** — deploy the new version only if it beats the currently-serving score (the same evaluation gate from Bonus A Q2), then keep collecting. Over time the thumbs-down examples that recur become the highest-signal backlog, so the agent improves on exactly the queries real users get wrong.

### Bonus C — Standalone MCP server (attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - **Gained:** *Independent scaling* — the tool service and the model scale to their own load (many agents can share one tool service instead of each bundling a copy). *Independent deployment/lifecycle* — you can fix or extend a tool and redeploy the MCP app without re-logging and re-deploying the model (and vice-versa), which also shrinks the model artifact and removes the fragile in-container stdio/subprocess handling that caused us the most grief. *Security surface* — the tools run in their own app with their own identity/permissions, so you can scope what they can touch separately from the model. *Observability* — the tool service has its own logs, metrics, and traces, so you can see tool traffic, latency, and errors on their own dashboard instead of buried in the model container.
   - **New failure modes:** *Network* — every tool call is now a remote HTTP round-trip that can fail, time out, or be slow (added latency vs. an in-process/stdio call); we saw this directly — with the app stopped, `load_mcp_tools()` fails. *Availability* — the model now has a runtime dependency on a second service; if the MCP app is down, scaling from zero, or mis-permissioned, the calculation steps fail even though the model itself is healthy. *Auth* — the remote endpoint needs its own authentication (a bearer token in our client), which is another credential to provision, rotate, and secure. *Consistency/versioning* — the tool service can now be a different version than the model expects, so the contract between them must be managed.
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - Layered defense: (a) **Require authenticated identity, not a shared secret.** Databricks Apps sit behind workspace auth, so require a valid token/OAuth on every request; better, give the serving endpoint a **service principal** and have the MCP app authorize *only that principal* (check the caller identity), so a leaked static token isn't enough. (b) **Least-privilege authorization** — the app grants access only to the endpoint's principal, not "any workspace user." (c) **Network isolation** — restrict the app so it's reachable only from within the workspace/VPC (private networking / IP allowlist) rather than the public internet, so even a stolen credential can't be used from outside. (d) **Transport security + hygiene** — HTTPS only, short-lived rotating tokens (from a secret scope, never hard-coded), and audit logging of who called which tool. The tools here are pure/deterministic math with no data access, so the main risk is abuse/DoS rather than data exfiltration — but the same principal-scoped, network-isolated pattern is what you'd insist on for tools that touch real data.
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - **Bundle (Part 1)** when the tools are small, stable, and used by a single model, when you want the fewest moving parts and no extra runtime dependency (one artifact deploys and scales atomically, no network hop, no second service to keep up), and when low, predictable latency matters. Our finance/math tools are exactly this profile, so bundling is the sensible default — the only cost is re-deploying the model to change a tool.
   - **Separate service (Bonus C)** when the tools are shared across many agents/models (deploy once, reuse everywhere), when they change on a different cadence than the model (independent release cycles), when they need their own scaling, security boundary, or observability, or when they're heavy/stateful (DB connections, secrets, large deps) that you don't want inflating every model image. The trade is real operational overhead — a second deployment, network reliability, and auth — so it pays off at scale or when the coupling in the bundled approach becomes the bottleneck, not for a single small agent.
