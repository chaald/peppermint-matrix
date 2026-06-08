---
name: code-review
description: Review code changes for logic correctness and coding style conformity. Use before committing, before PRs, or on explicit request.
---

# Code Review

Semi-automated code review using a subagent to review changes, followed by user confirmation before any implementation.

## Workflow

### 1. Determine Review Scope

Determine what to review based on context, in this order:

- **If user explicitly provides files or paths** — review those files (e.g. "review src/utils/config.py", "review main.py and hyperparameter_search.py")
- **If user says "review the project" or "review all"** — review the entire project tree
- **Default (no explicit scope)** — review the latest changes. Run `git diff HEAD~1 --stat` to find the most recent changes and review the diff (or full files if the diff is small)
- **If no prompt or chat history exists** — confirm with the user: "What would you like me to review?"

When the skill is triggered as part of another workflow (e.g. commit), the caller should provide the scope explicitly.

### 2. Spawn Review Subagent

Send a single `Task(subagent_type="general")` with:

**Prompt includes:**
- The scope: file paths, git diff, or project structure
- The **full content** of the STYLE_GUIDE.md (read from `.agents/skills/code-review/STYLE_GUIDE.md`)
- Explicit instructions:

```
You are a strict code reviewer. Your job is:

1. Review the provided code for LOGIC CORRECTNESS:
   - Off-by-one errors, edge cases (NaN, None, empty collections, 0.0 values)
   - Type mismatches between function signatures and callers
   - Unintended behavior from falsy checks (`if x` instead of `if x is not None`)
   - Race conditions or file-handle safety issues
   - Silent data loss (e.g. CSV fields that would be written but invisible to readers)
   - Missing error handling or unguarded operations

2. Review the provided code for CODING STYLE conformity (using the style guide below):
   - Import ordering violations
   - Function signature formatting violations (one param per line for multi-line)
   - DataFrame naming violations (no `df_` prefix)
   - NumPy array naming violations (use `_vector` suffix, not `_np`)
   - Inline comment violations (no `# ...` comments unless genuinely needed)
   - Missing docstrings for new functions/methods/classes
   - Naming convention mismatches

3. Provide findings in a structured format:
   🔴 CRITICAL: Bugs, logic errors, data integrity issues
   🟡 WARNING: Style violations, naming issues, robustness concerns  
   ⚪ SUGGESTION: Nice-to-have improvements, alternative approaches

Do NOT edit any files. Return your review report only.
```

### 3. Collect and Summarize

After the subagent returns:

- Group findings by severity (Critical → Warning → Suggestion)
- For each finding, include **the exact file path, line number, and current code** for context
- Exclude false positives or subjective opinions that don't match the style guide

### 4. Present to User

Present the categorized list cleanly. For example:

```
🔴 Critical (1):
  - config.py:750 — falsy guard drops 0.0 values: `if val else None` should be `if val is not None else None`

🟡 Warning (2):
  - main.py:56 — inline comment `# Set random seeds for reproducibility` violates no-inline-comment rule
  - hyperparameter_search.py:15 — import ordering: `from datetime` before stdlib group

⚪ Suggestion (1):
  - config.py:752 — `str(best_run_config)` could use `json.dumps` for downstream parseability
```

Then ask: **"Want me to fix these?"**. Do NOT implement until the user explicitly confirms.

### 5. Implement on Confirmation

When the user confirms:

- Fix critical items first, then warnings, then suggestions
- Fix one item at a time
- After each fix, verify with the user if the item had ambiguity
- Signal readiness for a follow-up review

## Scope Types

| Scope | Subagent Prompt Content |
|-------|------------------------|
| File(s) | Full file contents of each file |
| Git diff | `git diff HEAD~1` output + `git diff --stat` |
| Full project | File tree listing (`ls -R src/`) + key files |
