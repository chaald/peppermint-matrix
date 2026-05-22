# Configure members

Edit `council.config.yaml` to define council members:

```yaml
council:
  members:
    - name: analyst
      persona: >
        You are an analytical, detail-oriented thinker. Break down problems methodically.
      emoji: "🔍"

    - name: critic
      persona: >
        You are a critical reviewer. Find edge cases and risks.
      emoji: "⚡"

    - name: explorer
      persona: >
        You are a creative explorer. Think outside the box.
      emoji: "🌐"
```

Add custom members by appending entries to `members`:

- Use a stable `name` (lowercase, short).
- Set `persona` to the instruction injected into the subagent's prompt. This shapes the member's perspective and response style.
- Provide `emoji` for readability (optional but recommended).
- Optionally set `model` to a specific `provider/model` string (e.g., `anthropic/claude-sonnet-4-20250514`) to use a different model for that member. If not set, the current session model is used.
