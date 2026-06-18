# agentic

Self-contained agentic tool-use harness for bitnet.c. Loads a GGUF model,
runs 8 scripted turns with tool calls (datetime, file read/write/edit, glob,
grep, shell, diff), and writes a JSON transcript.

---

WARNING: this harness exposes exec_shell_command, write_file, edit_file, and
apply_diff. It runs whatever the model decides. Only use it inside a
disposable sandbox directory and never in an untrusted or production
environment.

---

## Prerequisites

- macOS or Linux
- A Qwen3.5 hybrid-SSM GGUF model (tokenizer.chat_template required)
- Engine objects already built in the repo root (see step 1)

## Build

**Step 1 - build the engine in the repo root:**

```
make bitnet
```

For Apple GPU (Metal):

```
make BN_ENABLE_METAL=1 bitnet
```

**Step 2 - build the harness:**

```
cd agentic && make
```

For Metal:

```
cd agentic && make BN_ENABLE_METAL=1
```

## Fetch a model

```
curl -L -o qwen3.5-4b.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
```

File size is approximately 2.7 GB.

## Run

Run from a throwaway sandbox directory so that file-writing tools do not
touch your real project:

```
mkdir sandbox && cd sandbox
BN_AGENT_TEMP=0 ../test_agent /abs/path/to/qwen3.5-4b.gguf --out tr.json
```

With Metal:

```
BN_AGENT_TEMP=0 ../test_agent /abs/path/to/qwen3.5-4b.gguf --out tr.json --metal
```

The transcript is written to `tr.json` (or the path given with `--out`).

## Environment variables

| Variable       | Default | Description                              |
|----------------|---------|------------------------------------------|
| BN_AGENT_TEMP  | 1.0     | Sampling temperature (0 = greedy)        |
| BN_AGENT_TOPP  | 1.0     | Top-p nucleus sampling threshold         |
| BN_AGENT_SEED  | 0       | RNG seed                                 |
