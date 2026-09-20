# Model Strategy

How agents use language models — **without hard-coding any specific LLM
or provider**. Design only; no provider chosen, no SDKs added.

## 1. The two models, never conflated

- **OpenCode's coding model (currently `big-pickle`)** — the agent that
  builds *this software*. It is a dev tool, not part of the runtime.
- **Production LLMs** — the models the YouTube-pipeline agents call
  (research, script, scene planning, fact-check assist).

They are completely separate. Nothing in the pipeline code references or
depends on the coding agent. Choosing billing/production providers is a
later decision, deliberately deferred.

## 2. Agents never depend on a concrete LLM

Agents depend on an **abstract model provider interface**. A warning in
the docs + interface shape keeps this honest.

Design sketch of the seam (not shipped code):

```python
# ModelProvider — the only thing agents import from the model layer
class ModelProvider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...

# ModelRequest
#   messages: str | list
#   system_prompt_ref: Optional[str]   # which versioned prompt to use
#   temperature: Optional[float]
#   max_tokens: Optional[int]

# ModelResponse
#   text: str
#   usage: tokens_in/out
#   provider_ref: str            # which provider+model actually ran
#   latency_ms: int
```

- `provider_ref` on every response makes artifact provenance possible.
- Adapters map this interface to each concrete provider later (hosted,
  self-hosted, local via Ollama etc.) — all swappable behind the same
  seam.

## 3. What the concrete choice must satisfy (constraint, not pick)

When a provider is eventually chosen it should be:

- cost / latency compatible with the channel's volume,
- capable of the instruction-following our structured-output prompts
  need (JSON extraction),
- usable from local-PC orchestration,
- auditable (model name + version reported per call).

Multiple providers can coexist per stage config; the interface makes
that a configuration detail, not a code change.

## 4. Prompts are versioned model inputs

`prompts/<stage>/` holds the prompt templates. Conventions:

- One file per versioned prompt (e.g. `research/facts-v1.txt`).
- Prompts are referenced by **name + version** in `ModelRequest`.
- Artifact headers record `produced_by.prompt_version` + `provider` +
  `model` (`DATA_CONTRACTS.md` §1), so any drift is diagnosable.
- Prompt changes are versioned like code; old runs remain reproducible.

## 5. Structured output discipline

Agents must return contract-shaped JSON. Approach (decision-friendly):

- enforce schema by asking the model for the contract shape, then
  validate with a schema validator (add the smallest dependency when
  implemented — likely the stdlib-adjacent choice first, revisit if
  painful).
- Never trust raw model output as a contract: validation happens before a
  stage's artifact is persisted as `completed`.

## 6. Where models are NOT allowed

- Not in the correctness path: research facts come from sources, the
  model only proposes (`RESEARCH_PIPELINE.md` §1).
- Not as the fact authority: fact check is rule-first, LLM-assist at most
  (`FACT_VERIFICATION.md` §4).
- Not in the stable worker contract: `production.json` contains zero
  model info (`DATA_CONTRACTS.md` §10).

## 7. Config placement

- Providers/models configured in `config/` + `.env` keys (no SDKs or
  credentials committed). `.env.example` already documents where keys
  land.

## 8. Open questions

- Validation dependency choice at implementation time (stdlib-only
  draft vs. a schema library).
- JSON extraction strategy (prompt-only vs. structured decoders
  depending on provider support).
- Retry/backoff and token-budget policy per stage (tied to model
  choices).