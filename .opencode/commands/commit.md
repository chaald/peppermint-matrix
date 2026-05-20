---
description: Stage and commit changes with proper message
agent: general
subtask: true
---

Stage and commit the current changes in this repo. Do NOT push.

1. Run `git status`, `git diff --stat`, `git diff --cached --stat`, and `git log --oneline -5` to understand the current state — some files may already be staged.
2. If there are unstaged changes, stage them with `git add`. Leave already-staged files as-is.
3. Craft a commit message in the existing project style (check recent commits for the `[foryou]` prefix pattern).
4. Run `git commit` with the message.
5. Verify with `git status` that the working tree is clean.

The commit message should be concise (1 line summary, no body unless the changes span multiple distinct areas).
