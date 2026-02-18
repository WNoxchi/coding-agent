# Spec: Terminal Chat Interface for `coder.go`

## Status
Approved baseline for initial implementation.

## Scope
Build a minimal interactive terminal chat interface in `coder.go`.

- In scope: Anthropic-backed chat loop, terminal formatting, verbose logging, startup/config behavior.
- Out of scope: Any files under `external/*`, advanced tool execution engine, multi-provider routing, persistence.

## Why This Exists
You are building a coding agent incrementally. The first usable milestone is a reliable local CLI chat loop that:

1. Authenticates with Anthropic from environment.
2. Sends user prompts to a default Claude model.
3. Clearly distinguishes `User` vs `Claude` output in the terminal.
4. Exposes internal operations via verbose logs so future agent/tool behavior is debuggable.

This creates a stable foundation for later features (tools, memory, planning, etc.) without over-designing the first step.

In this document, `v1` means the first shippable implementation milestone of `coder.go` (the baseline chat loop described here).

## User Experience

### Startup
- Command: `go run coder.go [flags]`
- On start:
  - Read `ANTHROPIC_API_KEY` from environment.
  - If missing/empty: print actionable error and exit non-zero.
  - Default model must be Claude Sonnet 4.6 with model ID `claude-sonnet-4-6`.

### Interactive loop
- Prompt format for user input prefix:
  - Text: `User: `
  - Color: light blue
- Assistant output prefix for Anthropic models:
  - Text format: `Claude (<model display name>): `
  - Example: `Claude (Sonnet 4.6): `
  - Color: Anthropic-like orange
- For each submitted user prompt:
  - Send prompt to Anthropic using the configured model and API key.
  - Print assistant text response.
- Loop continues until user exits.

### Exit behavior
- Exit on `Ctrl+C`, `Ctrl+D`, or explicit quit command (`/quit` and `/exit`).
- Exit should be clean (no stack traces on normal termination).

## Functional Requirements

### Configuration
- Required env var:
  - `ANTHROPIC_API_KEY`
- CLI flags:
  - `--verbose` (or `-v`): enable detailed operation logging.
  - `--model <id>`: override model ID.
- Model default:
  - display name: Claude Sonnet 4.6
  - API model ID: `claude-sonnet-4-6`

### API behavior
- Use Anthropic Messages API.
- For MVP, send a simple message payload with user text.
- Maintain conversation history in-memory for natural multi-turn chat.
- Response rendering mode for this milestone is non-streaming:
  - print assistant output once the API response is complete.
- Handle API errors gracefully:
  - Print concise user-facing error.
  - Continue REPL unless error is unrecoverable.

### Verbose mode behavior
When enabled, print structured diagnostics for every significant operation, including:

1. Startup config resolution:
   - model selected
   - API key presence (only boolean/length, never raw key)
2. Per-turn input handling:
   - prompt received (or prompt length if redaction preferred)
3. API call lifecycle:
   - request start timestamp
   - model used
   - success/failure
   - latency
   - response metadata available from SDK (request id, usage tokens if available)
4. Tool-related response items:
   - if model returns tool-use content blocks, log block names/ids/inputs
   - if no tool blocks are present, log that none were returned

Verbose logs should go to `stderr` so normal chat remains readable on `stdout`.

Verbose logs should be summarized structured events, not full raw payload dumps by default. Example shape:

```text
ts=... level=info event=api_request_start model=claude-sonnet-4-6 turn=3
ts=... level=info event=api_request_end ok=true latency_ms=842 input_tokens=120 output_tokens=312 request_id=...
```

### Terminal formatting
- Use ANSI color escapes for prefixes only (reset after prefix).
- Suggested colors:
  - User light blue: ANSI truecolor `38;2;102;178;255`
  - Claude orange: ANSI truecolor `38;2;217;119;6`
- If color output is unsupported, degrade to plain text automatically.

## Non-Functional Requirements
- Keep implementation small and readable; this is a foundation layer.
- Never log secrets (API key, auth headers).
- Clear error messages for common failures (missing key, network failure, rate limits).
- No dependency on `external/*`.

## Proposed Program Structure (`coder.go`)

- `main()`
  - parse flags
  - load config
  - build client
  - start chat loop
- `type Config`
  - `APIKey string`
  - `Model string`
  - `Verbose bool`
- `runChatLoop(cfg Config, client ...) error`
  - read stdin line-by-line
  - append to conversation
  - call model
  - render output
- `sendAnthropicMessage(...)`
  - translate conversation to API format
  - execute call
  - return assistant text + metadata
- `logVerbose(event string, kv ...any)`
  - centralized verbose logging with timestamps
- `formatUserPrefix()`, `formatAssistantPrefix(model string)`
  - colorized prefix formatting

## Acceptance Criteria

1. Running without `ANTHROPIC_API_KEY` exits with a clear message and non-zero code.
2. Running with key starts interactive prompt and displays `User: ` in light blue.
3. Submitting text makes an Anthropic API request using default Sonnet 4.6 model.
4. Assistant response prints with prefix `Claude (Sonnet 4.6): ` in orange.
5. `--verbose` prints operation logs for API call start/end, model used, success/failure, and response metadata.
6. API errors are reported without crashing the process (unless unrecoverable init failure).
7. `/quit`, `/exit`, `Ctrl+C`, and `Ctrl+D` terminate cleanly.

## Manual Test Plan

1. Missing key test:
   - Unset `ANTHROPIC_API_KEY`
   - Run `go run coder.go`
   - Expect explicit error and exit code `!= 0`
2. Happy path test:
   - Set valid API key
   - Run `go run coder.go`
   - Enter prompt like `hello`
   - Expect assistant response with correct colored prefix
3. Verbose test:
   - Run `go run coder.go --verbose`
   - Send prompt
   - Expect detailed logs on `stderr` including model and timing
4. Exit test:
   - Confirm `/quit`, `/exit`, `Ctrl+C`, `Ctrl+D` all exit cleanly
