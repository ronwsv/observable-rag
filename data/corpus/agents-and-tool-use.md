# Agents and Tool Use

An **agent** is an LLM that can decide to take actions — call tools, query a
retriever, or hand off — rather than only emitting text. Tool use (also called
function calling) lets the model choose a function and produce structured arguments
for it, which the application then executes.

## Query routing

Not every question needs retrieval. A greeting ("hi"), a request to rephrase the
previous answer, or a pure arithmetic question can be answered directly, while a
factual question about the knowledge base should trigger retrieval. A **router** is
a lightweight decision step that classifies the query and picks an action:

- `retrieve` — search the knowledge base, then answer from the retrieved context.
- `direct` — answer directly without retrieval (chit-chat, formatting, reasoning).

Routing saves latency and cost by skipping retrieval when it adds nothing, and it
improves quality by not stuffing irrelevant context into simple requests.

## Function calling

Modern LLM APIs accept a list of tool definitions (name, description, JSON schema
for arguments). The model returns either a normal message or a structured tool call.
The application runs the tool and feeds the result back to the model. Robust tool
use requires validating the model's arguments against the schema before executing.

## Keeping agents reliable

- Constrain the action space: fewer, well-described tools are more reliable than
  many overlapping ones.
- Validate every tool call's arguments; never execute unchecked model output.
- Log each decision (which tool, which arguments, what result) so agent behavior is
  observable and testable.
- Add a fallback: if routing or a tool fails, degrade gracefully to a plain answer
  rather than erroring out.

## Agentic RAG

Combining routing with retrieval yields "agentic RAG": the model decides whether to
retrieve, can retrieve more than once, and can reformulate the query if the first
retrieval was weak. This is more powerful than a fixed pipeline but must be measured
with the same retrieval and faithfulness metrics to avoid unbounded, expensive loops.
