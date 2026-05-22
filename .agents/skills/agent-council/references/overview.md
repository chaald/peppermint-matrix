# Overview

- Gather responses from configured council members via OpenCode subagents (Task tool).
- The chairman (current OpenCode session) synthesizes the final response.
- Configure members in `council.config.yaml`.
- Reference [Karpathy's LLM Council](https://github.com/karpathy/llm-council) for inspiration.

## Workflow (3 stages)

1. Spawn one OpenCode subagent per configured member, each with the user's question and the member's persona instruction. Send all subagent calls in parallel.
2. Collect and surface all member responses.
3. The chairman synthesizes the final answer: compare perspectives, highlight agreements and disagreements, and provide a unified recommendation.
