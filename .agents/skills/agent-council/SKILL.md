---
name: agent-council
description: Collect and synthesize opinions from multiple AI agents. Use when users say "summon the council", "ask other AIs", or want multiple AI perspectives on a question.
---

# Agent Council

Collect multiple AI perspectives and synthesize one answer using OpenCode's native subagent system.

## Workflow

When the user asks to "summon the council" or requests multiple AI opinions:

1. Read `council.config.yaml` to get the configured members and their personas.
2. Spawn one OpenCode subagent per member using the `Task` tool. Send all subagent calls in a single message (parallel). Each subagent call:
   - Uses `subagent_type: general`.
   - Receives the user's original question plus the member's `persona` instruction from the config as the prompt.
   - If the member has a `model` field, request that model.
3. Collect all subagent responses once they complete.
4. Synthesize the final answer as chairman: compare perspectives, identify consensus and disagreements, and provide a unified recommendation. Surface individual member opinions before the synthesis.

## Configuration

Edit `council.config.yaml` to add, remove, or modify council members. Each member needs:
- `name` — unique identifier.
- `persona` — instruction injected into the subagent's prompt (e.g. "You are an analytical thinker...").
- `emoji` — display emoji (optional).
- `model` — OpenCode model string in `provider/model` format (optional; uses current session model if not set).

## Alternative: Shell Scripts (Legacy)

The `scripts/` directory contains the original shell-script-based job runner that spawns external CLI tools. It is preserved for reference but not needed when using the OpenCode-native Task-tool workflow above.

## References

- `references/overview.md` — workflow and background.
- `references/examples.md` — usage examples.
- `references/config.md` — member configuration.
- `references/requirements.md` — dependencies.
- `references/safety.md` — safety notes.
