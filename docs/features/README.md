# Feature Documentation

This folder contains documentation for features being developed in this project.

## Structure

Each feature gets its own `.md` file:

```
docs/features/
  README.md          ← this file (index)
  feature-name.md    ← one file per feature
```

## Feature Index

| Feature | File | Date Added | Status |
|---------|------|------------|--------|
| Surrogate Model & Exploration Saturation Metric | [surrogate-model.md](surrogate-model.md) | 2026-03-15 | Implemented in notebook |
| Model-Based Hyperparameter Search | [model-based-hyperparameter-search.md](model-based-hyperparameter-search.md) | 2026-03-15 | Planned |
| Jupyter MCP Local Setup | [jupyter-mcp.md](jupyter-mcp.md) | 2026-04-24 | Configured locally |
| Convergence Simulation | [convergence-simulation.md](convergence-simulation.md) | 2026-05-22 | Planned |

## Template

Use this as a starting point for new feature docs:

```markdown
# Feature Name
**Date:** YYYY-MM-DD

## Overview
Brief description of what this feature does.

## Motivation
Why this feature is needed.

## Design
How it works / key implementation decisions.

## Usage
How to use it (code examples if applicable).

## Status
- [ ] In progress / completed / planned
```
