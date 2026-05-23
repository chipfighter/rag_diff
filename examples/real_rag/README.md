# Example: a real RAG pipeline for rag-diff

A small but real RAG system used to prove that rag-diff catches and correctly
diagnoses regressions. It indexes a fake company HR handbook (`corpus/`),
retrieves with **local embeddings** (free, offline), and answers with
**DeepSeek**.

## Setup

From the repo root:

```powershell
pip install -e ".[openai,dev]"
pip install -r examples/real_rag/requirements.txt
```

Copy `env.example` (at the repo root) to `.env` and add your DeepSeek key:

```
OPENAI_API_KEY=your-deepseek-api-key
OPENAI_BASE_URL=https://api.deepseek.com
```

Any OpenAI-compatible API works — just change the key, base URL, and
`--judge-model`.

## Regression knobs

The adapter reads two environment variables so you can break it on purpose:

| Variable          | Default     | Effect                                              |
|-------------------|-------------|-----------------------------------------------------|
| `RAG_TOPK`        | `4`         | Number of chunks retrieved.                         |
| `RAG_RETRIEVAL`   | `good`      | Set to `bad` to return the least-relevant chunks.   |
| `RAG_PROMPT_MODE` | `grounded`  | Set to `lazy` to make the model ignore the context. |

## Retrieval-only check (no API key needed)

```powershell
python examples/real_rag/pipeline.py
```

## Run the proof

```powershell
$T = "examples/real_rag/testset.json"
$A = "examples/real_rag/adapter.py:ask"

# 1. Good baseline run. Note the run id it prints, e.g. run_2026..._abcd1234
rag-diff run --testset $T --target $A --judge-model deepseek-chat

# 2. Retrieval regression -> expect "Suspected Retrieval Issue"
$env:RAG_RETRIEVAL = "bad"
rag-diff run --testset $T --target $A --judge-model deepseek-chat --compare-to <BASE_RUN_ID>
Remove-Item Env:\RAG_RETRIEVAL

# 3. Model regression -> expect "Suspected Model Issue"
$env:RAG_PROMPT_MODE = "lazy"
rag-diff run --testset $T --target $A --judge-model deepseek-chat --compare-to <BASE_RUN_ID>
Remove-Item Env:\RAG_PROMPT_MODE
```

Replace `<BASE_RUN_ID>` with the id from step 1 so both regressions are compared
against the same good baseline.
