---
name: commit
description: Use when the user wants to stage and commit changes, typically saying "let's commit", "commit first", "commit changes", "time to commit", or similar
---

# Commit

Stage and commit current changes without pushing.

1. Run `git status`, `git diff --stat`, `git diff --cached --stat`, and `git log --oneline -5` to understand the state
2. By default, stage all unstaged changes with `git add`. Only skip this if the user explicitly said to commit only staged files.
3. Craft a commit message in the existing project style
4. Run `git commit` with the message
5. Verify with `git status` that the working tree is clean

Message should be concise (1 line summary; no body unless changes span multiple distinct areas).
