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

## Running locally

TODO: how to ingest the corpus, run the graph in `pa4.ipynb`, and test queries.

> **Example of the level of detail expected** (replace with your own steps/values):
>
> 1. **Ingest the corpus** (run once, from a Databricks notebook):
>    ```python
>    from rag.ingest import ingest
>    ingest(spark, volume_path="/Volumes/main/default/pa4/annual_report.pdf")
>    ```
>    This parses the PDF, chunks it into `main.default.ali_analyst_chunks`, and syncs the
>    Vector Search index `main.default.ali_analyst_index`. Wait until the index is `READY`.
>
> 2. **Build and run the graph** in `pa4.ipynb`:
>    ```python
>    from agent.graph import build_graph
>    graph = build_graph()          # uses config.py + rag/store.py + the MCP server
>    result = graph.invoke({"messages": [{"role": "user",
>              "content": "What was the net revenue in 2023?"}]})
>    print(result["messages"][-1].content)
>    ```
>
> 3. **Test queries I ran** (retrieval-only, computation-only, combined):
>    | Query | Answer produced |
>    |-------|-----------------|
>    | "What was the net income in 2023?" | ¥1.11 trillion [source: annual_report.pdf, p.4] |
>    | "What is 15% of 2.4 billion?" | 360 million |
>    | "What was 2023 revenue, and its value after 10% growth?" | ¥16.91T → ¥18.60T (16.91 × 1.10) |

## Deployment

TODO: how you logged, registered, and served the model; endpoint name; URL.

## Design decisions

TODO: graph architecture, routing, deployment choices.

---

## Analysis Questions

> Answer in your own words. Each question is copied from the assignment so you don't have to flip back.

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   - It will be handled automatically becasue the steps execute sequentially. Step 3 can use result step 1 because the results are stored in `step_results` list.
2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   - It will hurt the perofrmance because it adds the latency if we replan after each step. For our current work, planning once is enough.

### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
   - If the supervisor misroutes a step say it routes a calculation step to rag tool. It will fetch irrevlant details or return "Not found". This step will fail silently without crashing. We can detect this by having out tools tag their output type and letting supervisor re-classify the step if the results looks off topic
2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   - In a single ReAct pattern, the agent has access to all the tools and it can call the tools freely itself without relying on anyone, it is much simpler than supervisor pattern. In projects, where we have hetrogenous tasks that requires completely different tools from each other, supervisor is much more effective and delegate the tasks efficiently to the relevant tools.

### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   - TODO
2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   - TODO

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   - TODO
2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - TODO

### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   - TODO
2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   - TODO

### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   - TODO
2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   - TODO
3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   - TODO

### Bonus A — CI/CD (if attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - TODO
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - TODO

### Bonus B — `databricks-agents` SDK (if attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - TODO
2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - TODO

### Bonus C — Standalone MCP server (if attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - TODO
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - TODO
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - TODO
