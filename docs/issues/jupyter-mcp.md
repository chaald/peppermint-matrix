# Jupyter MCP Collaboration Instability

## Summary

The `jupyter_server_ydoc` (jupyter-collaboration) extension, which powers MCP notebook operations (`use_notebook`, `read_notebook`, `insert_cell`, etc.), enters an unrecoverable state where a specific notebook permanently reports 0 cells. This state survives server restarts, database deletion, and file re-creation.

## Root Cause

The instability has two compounding causes:

### 1. File Watcher Error Suppression (`rooms.py:53-59`)

`file_stop_poll_on_errors_after` defaults to **24 hours** (86400 seconds):

```python
file_stop_poll_on_errors_after = Float(
    24 * 60 * 60,
    allow_none=True,
    config=True,
    help="""The duration in seconds to stop polling a file after consecutive errors.
    Defaults to 24 hours, if None then polling will not stop on errors.""",
)
```

When `convergence_simulation.ipynb` was briefly renamed away (to `_old.ipynb`), the file watcher received repeated HTTP 404 errors. After `file_stop_poll_on_errors_after` seconds of continuous errors, the watcher suppressed all future monitoring for that file path — even after the file was restored.

### 2. YStore Room Persistence (`stores.py:22-28`)

`jupyter_server_ydoc` uses `pycrdt.store.SQLiteYStore` to persist Yjs document rooms in `.jupyter_ystore.db`:

```python
class SQLiteYStore(LoggingConfigurable, _SQLiteYStore):
    db_path = Unicode(
        ".jupyter_ystore.db",
        config=True,
        help="""The path to the YStore database.""",
    )
```

The room (`e820f5be`) was initialized with an empty 0-cell document when the file was absent. This stale state was persisted to `.jupyter_ystore.db`. Deleting the database alone didn't help because the Yjs room was re-created with the same empty state by the file watcher's error-suppressed polling.

### 3. Init Priority (`rooms.py:111-164`)

`DocumentRoom.initialize()` loads from YStore **before** disk. When the YStore has a saved state (even an empty one), the room treats it as authoritative and never re-reads from disk. The out-of-sync check at line 143 compares content but doesn't force a reload when disk has content and YStore is empty.

## Symptoms

- **Only one notebook affected**: `convergence_simulation.ipynb` permanently showed 0 cells. All other notebooks (`oracle_model.ipynb`, `surrogate_model.ipynb`) worked fine via MCP.
- **Contents API works**: The standard Jupyter REST API at `/api/contents/...` correctly returned the full notebook content (verified 3 cells).
- **Write operations work**: MCP `insert_cell`, `delete_cell`, and `overwrite_cell_source` all functioned correctly on the 0-cell notebook.
- **Read operations broken**: MCP `read_notebook` and `use_notebook` always showed 0 cells.
- **Survived all recovery attempts**: Server restart, `.jupyter_ystore.db` deletion, `file_id_manager.db` deletion, file deletion and re-creation, kernel restart, full server rebuild.

## Affected User Flow

The MCP MCP tools cannot auto-discover kernels because they depend on the collaboration extension's session API. Each conversation requires manual kernel ID handoff:

1. User opens notebook in JupyterLab.
2. User runs the import cell.
3. User copies kernel ID from the kernel output.
4. User sends kernel ID to the agent.
5. Agent calls `use_notebook(mode="connect", kernel_id=...)`.

## Plan: Custom MCP Server (`jupyter-direct`)

A new standalone MCP server that bypasses the collaboration extension entirely.

### Architecture

```
Agent ←stdio→ jupyter-direct MCP server ←ZMQ→ kernel
                                          ←HTTP→ Jupyter Contents API
```

### Tools

| Tool | Implementation | Why reliable |
|------|---------------|--------------|
| `execute_code` | ZMQ `execute_request` via `jupyter_client.KernelClient` | Same protocol as JupyterLab — no browser, no Yjs |
| `read_notebook` | `GET /api/contents/...` REST API | Already verified working independently |
| `write_notebook` | Python `json.dump` to `.ipynb` file | Direct disk I/O |
| `list_kernels` | Parse `~/.local/share/jupyter/runtime/kernel-*.json` | Direct file system — no API dependency |
| `list_notebooks` | `jupyter list` or filesystem scan | Standard discovery |

### What is eliminated

- `jupyter-collaboration` / `jupyter_server_ydoc`
- Yjs document rooms and `pycrdt`
- `SQLiteYStore` / `.jupyter_ystore.db`
- WebSocket connection to JupyterLab frontend
- `docmanager_open` and `DocSessionHandler`
- Corrupted 0-cell states
- Manual kernel ID handoff

### Implementation Notes

- Uses `jupyter_client` (already in `.venv`) for ZMQ kernel communication.
- A kernel's connection info is stored at `~/.local/share/jupyter/runtime/kernel-{id}.json`. Auto-discovery scans these files and tests connectivity.
- The Contents REST API is stable and tested — `/api/contents/{path}` returns full notebook JSON.
- Estimated scope: ~300 lines in a single file `mcp_server.py`.

### Risiko

- Jupyter's ZMQ protocol version mismatch — mitigated by `jupyter_client` which handles protocol negotiation.
- The MCP SDK (Python) is relatively new (~0.1.x). If incompatible, the server could use stdio JSON-RPC directly instead.
- No real-time cell output streaming yet. The first version uses `execute_code` with polling, which is sufficient for our use case.
