# Jupyter MCP Local Setup
**Date:** 2026-04-24

## Overview

This repo supports a local Jupyter MCP setup for OpenCode through `opencode.json` and a local JupyterLab instance started from the repo `.venv`.

## Motivation

The goal is to let OpenCode create notebooks, insert cells, edit cells, and execute notebook content without manual browser interaction.

## Design

- OpenCode loads `opencode.json` and starts `jupyter-mcp-server` with `uvx`.
- JupyterLab runs locally on `127.0.0.1:5601` and uses the repo venv.
- The MCP stack reads `JUPYTER_URL`, `JUPYTER_TOKEN`, and `MCP_TOKEN` from `.env`.
- Notebook collaboration features depend on `jupyterlab==4.5.6`, `notebook==7.5.5`, `jupyter-collaboration==4.3.0`, `jupyter-mcp-tools>=0.1.4`, and `datalayer-pycrdt==0.12.17`.

## Usage

1. Copy `.env.example` to `.env` and fill in the token values.
2. Load the env file before starting OpenCode or JupyterLab:

```bash
set -a; source .env; set +a
```

3. Start JupyterLab from the repo venv:

```bash
.venv/bin/jupyter lab --no-browser --port=5601 --ip=127.0.0.1 --IdentityProvider.token="$JUPYTER_TOKEN"
```

4. Start OpenCode in the same shell so the MCP server inherits the env vars.
5. If collaboration-based cell edits fail on `/api/collaboration/session/...`, restart JupyterLab and reconnect the notebook through MCP.

## Status

- [x] Configured locally for OpenCode
