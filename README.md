<p align="center">
  <img src="https://raw.githubusercontent.com/juyterman1000/entroly/main/docs/assets/logo.png" width="180" alt="Entroly — Save 80-95% on LLM Tokens">
</p>

<h1 align="center">Entroly</h1>

<h3 align="center">Cut your AI token costs by 80–95%. Zero accuracy loss.</h3>

<p align="center">
  <b>The Token Optimization Engine for AI Coding</b><br/>
  <i>Drop-in proxy between your IDE and any LLM API. Same answers, 80–95% fewer tokens, 10x lower bills.</i>
</p>

<p align="center">
  <code>pip install entroly && entroly go</code>
</p>

<p align="center">
  <a href="#token-savings-at-a-glance">Savings</a> &bull;
  <a href="#30-second-install">Install</a> &bull;
  <a href="#how-we-save-tokens">How It Works</a> &bull;
  <a href="#works-with-everything">Integrations</a> &bull;
  <a href="#track-your-savings">Dashboard</a> &bull;
  <a href="https://github.com/juyterman1000/entroly/discussions">Community</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/entroly"><img src="https://img.shields.io/pypi/v/entroly?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/entroly"><img src="https://img.shields.io/npm/v/entroly?color=red&label=npm" alt="npm"></a>
  <img src="https://img.shields.io/badge/Token_Savings-80--95%25-success" alt="Savings">
  <img src="https://img.shields.io/badge/Latency-<10ms-purple" alt="Latency">
  <img src="https://img.shields.io/badge/Tests-840_Passing-success" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

---

## Token Savings at a Glance

| | Without Entroly | With Entroly | You Save |
|---|---|---|---|
| **Tokens per request** | 186,000 | 9,300–37,000 | **80–95%** |
| **Cost per 1K requests (GPT-4o)** | ~$560 | ~$28–$112 | **$450+** |
| **Monthly cost (100 req/day)** | ~$1,700 | ~$85–$340 | **$1,300+/mo** |
| **Token waste (duplicates, boilerplate)** | ~40% | **0%** | Eliminated |
| **Quality** | Partial, hallucinated | **Same or better** | ↑ |

**Every token we cut is a token that wasn't helping the LLM anyway.** Duplicates, boilerplate, stale context — gone. The AI gets the same information in fewer tokens.

---

## 30-Second Install

```bash
pip install entroly[full]
entroly go
```

**That's it.** Point your AI tool to `http://localhost:9377/v1`. Every request is now optimized.

```bash
# Or step by step
pip install entroly                # core engine
entroly init                       # detect IDE + generate config
entroly proxy --quality balanced   # start the token optimizer
```

### Other Installs

```bash
npm install entroly                # Node.js
docker pull ghcr.io/juyterman1000/entroly:latest   # Docker
```

| Package | What you get |
|---------|---|
| `pip install entroly` | Core — MCP server + Python engine |
| `pip install entroly[proxy]` | + HTTP proxy (token optimization) |
| `pip install entroly[native]` | + Rust engine (50-100x faster) |
| `pip install entroly[full]` | Everything |

---

## How We Save Tokens

Entroly sits between your AI tool and the LLM API. It compresses everything — codebase context, conversation history, tool outputs — before tokens are spent.

### 1. Context Compression (saves 70–95%)

Your codebase has thousands of files. Raw-dumping them wastes tokens on duplicates, boilerplate, and irrelevant code.

Entroly indexes your codebase and selects the **mathematically optimal** subset for each query:
- Critical files → **full code**
- Supporting files → **signatures only** (60–80% smaller)
- Everything else → **one-line references**

The AI sees your entire codebase. You pay for a fraction of it.

### 2. Conversation Pruning (saves 50–80% on history)

In a 50-turn conversation, old messages are re-sent on every request — wasting 20–40K tokens. Entroly compresses old messages automatically:
- Last 2 turns → full fidelity
- 3–5 turns back → key decisions + code only
- Older → summarized to ~20% of original

### 3. Tool Output Compression (saves 60–90%)

LLM tool calls return massive outputs — 2000-line JSON responses, test logs, git diffs. Entroly compresses them before they hit the LLM:

| Output Type | What Entroly Does | Savings |
|---|---|---|
| JSON blobs | Schema only (strip values) | ~85% |
| Test output | Failures only | ~90% |
| Build errors | Errors + warnings only | ~80% |
| Git diff | Compact hunks | ~70% |
| Log output | Deduplicated | ~75% |

### 4. Self-Correcting Quality (zero accuracy loss)

Entroly monitors LLM responses for confusion signals. If the AI says "I don't have enough context," Entroly automatically shifts toward less-compressed representations — **without increasing the token budget**. Same cost, better information density.

---

## Track Your Savings

```bash
entroly dashboard    # live web dashboard
entroly status       # CLI summary
entroly digest       # weekly report
entroly share        # shareable Context Report Card
```

The dashboard tracks **hourly, daily, weekly, and monthly** token savings with cost estimates per model.

---

## Works With Everything

| AI Tool | Setup | Method |
|---------|-------|--------|
| **Cursor** | `entroly init` | MCP server |
| **Claude Code** | `claude mcp add entroly -- entroly` | MCP server |
| **VS Code + Copilot** | `entroly init` | MCP server |
| **Windsurf** | `entroly init` | MCP server |
| **Cline** | `entroly init` | MCP server |
| **Cody** | `entroly proxy` | HTTP proxy |
| **Any LLM API** | `entroly proxy` | HTTP proxy |

Works with **OpenAI, Anthropic, Google Gemini, Azure OpenAI, and any OpenAI-compatible API**.

---

## SDK — 3 Lines

```python
from entroly import compress

optimized = compress(my_context, budget=4096)
# 80-95% fewer tokens, same information
```

Or use with LangChain:

```python
from entroly.integrations.langchain import EntrolyCompressor

chain = EntrolyCompressor(budget=8192) | llm
```

---

## CLI

| Command | What it does |
|---------|---|
| `entroly go` | **One command** — auto-detect, init, proxy, dashboard |
| `entroly demo` | Before/after comparison with dollar savings on YOUR project |
| `entroly dashboard` | Live metrics: savings trends, daily/weekly/monthly charts |
| `entroly doctor` | 7 diagnostic checks |
| `entroly health` | Codebase health grade (A–F): clones, dead code, god files |
| `entroly benchmark` | Benchmark: Entroly vs raw context vs top-K |
| `entroly autotune` | Auto-optimize engine parameters |
| `entroly share` | Generate a shareable Context Report Card |
| `entroly digest` | Weekly summary: tokens saved, cost reduction |
| `entroly status` | Check running services |

---

## Platform Support

| | Linux | macOS | Windows |
|--|---|---|---|
| **Python 3.10+** | ✅ | ✅ | ✅ |
| **Rust wheel** | ✅ | ✅ (Intel + Apple Silicon) | ✅ |
| **Docker** | Optional | Optional | Optional |
| **Admin/WSL required** | No | No | No |

---

## Production Ready

- **Persistent savings tracking** — lifetime savings in `~/.entroly/value_tracker.json`
- **Daily/weekly/monthly trend charts** in the dashboard
- **Rich response headers** — `X-Entroly-Tokens-Saved-Pct`, `X-Entroly-Cost-Saved-Today`
- **Crash recovery** — gzipped checkpoints restore in <100ms
- **55 SAST rules** — catches hardcoded secrets, SQL injection across 8 CWE categories
- **840 tests** — 399 Rust + 441 Python, CI verified

---

## How is Entroly Different from RAG?

| | RAG (vector search) | Entroly |
|--|---|---|
| **What it sends** | Top-K similar chunks | **Entire codebase** at optimal resolution |
| **Duplicates** | Sends same code 3x | **Eliminated** (SimHash dedup) |
| **Dependencies** | No | **Auto-includes** related files |
| **Learns** | No | **Yes** — RL optimizes from response quality |
| **External API needed** | Yes (embeddings) | **No** — runs locally |
| **Selection** | Approximate | **Mathematically proven** (knapsack solver) |

---

<details>
<summary><b>Technical Deep Dive — Architecture & Algorithms</b></summary>

### Architecture

Hybrid Rust + Python. All math in Rust via PyO3 (50–100x faster). MCP + orchestration in Python.

### Rust Core (21 modules)

| Module | What |
|---|---|
| **knapsack.rs** | Token-optimal context selection (KKT dual bisection) |
| **knapsack_sds.rs** | Submodular diversity + multi-resolution knapsack |
| **skeleton.rs** | Code skeletons — signatures only (60–80% token reduction) |
| **dedup.rs** | Duplicate detection — 64-bit SimHash |
| **lsh.rs** | Semantic recall — 12-table multi-probe LSH |
| **prism.rs** | Weight optimizer — spectral natural gradient |
| **entropy.rs** | Information density scoring |
| **depgraph.rs** | Dependency graph — auto-links imports and type refs |
| **sast.rs** | Security scanning — 55 rules, 8 CWE categories |
| **cache.rs** | EGSC cache with DAG-aware eviction |
| **causal.rs** | Causal context graph with do-calculus |
| **query_persona.rs** | Query archetypes via RBF kernel + Pitman-Yor process |
| **utilization.rs** | Response utilization feedback |
| **resonance.rs** | Context resonance matrix + fragment consolidation |

### References

Shannon (1948), Charikar (2002), Nemhauser-Wolsey-Fisher (1978), Sviridenko (2004), Boyd & Vandenberghe (Convex Optimization), LLMLingua (EMNLP 2023), RepoFormer (ICML 2024).

</details>

---

## Need Help?

```bash
entroly doctor    # runs 7 diagnostic checks
entroly --help    # all commands
```

**Email:** autobotbugfix@gmail.com

<details>
<summary><b>Common Issues</b></summary>

**macOS "externally-managed-environment":**
```bash
python3 -m venv ~/.venvs/entroly && source ~/.venvs/entroly/bin/activate && pip install entroly[full]
```

**Windows pip not found:**
```powershell
python -m pip install entroly
```

**Port 9377 in use:**
```bash
entroly proxy --port 9378
```

</details>

---

## Frequently Asked Questions

### How do I reduce my OpenAI / Claude / Gemini API costs?

Install Entroly. It sits between your app and the LLM API, compressing tokens by 80–95% before they're sent. You get the same answers at a fraction of the cost. Works with **OpenAI GPT-4o, Claude Opus, Claude Sonnet, Gemini Pro, and any OpenAI-compatible API**. One command: `pip install entroly && entroly go`.

### How do I save tokens on LLM API calls?

Entroly automatically compresses your codebase context, conversation history, and tool outputs — the three biggest sources of token waste. Average savings: **80–95% fewer tokens** per request with zero accuracy loss. It's a drop-in HTTP proxy — no code changes needed.

### What's the best LLM token optimizer?

Entroly uses mathematically proven algorithms (knapsack optimization, submodular maximization, information-theoretic compression) to select the optimal context for each query. It's the only tool that compresses context, conversation history, AND tool outputs together. 840 tests, Rust-powered, <10ms latency.

### How do I reduce token usage in Cursor / Claude Code / Copilot?

Run `entroly go` in your project directory. Entroly auto-detects your IDE, indexes your codebase, and starts optimizing every API request. No configuration needed. Works with Cursor, Claude Code, VS Code + Copilot, Windsurf, Cline, and any tool that calls an LLM API.

### Is there a free tool to compress LLM context?

Yes. Entroly is **open source (MIT)** and completely free. `pip install entroly && entroly go`. It compresses your codebase context by 80–95% using information-theoretic algorithms — no external APIs, no embeddings, everything runs locally on your machine.

### How do I spend less on AI coding assistants?

Most of your AI bill comes from redundant tokens — duplicate code, stale conversation history, verbose tool outputs. Entroly eliminates all three. Typical savings: **$1,300+/month** for teams making 100 requests/day.

---

## License

MIT

---

<p align="center">
  <b>Stop burning tokens. Save 80–95% on every LLM call.</b><br/>
  <code>pip install entroly && entroly go</code>
</p>

<!-- SEO: entroly, save llm tokens, reduce openai costs, reduce claude costs, reduce gemini costs, token optimization, llm token compression, cheaper ai api calls, reduce ai api costs, save money on ai, token savings, context compression, reduce anthropic costs, llm cost reduction, ai token optimizer, compress llm context, fewer tokens same quality -->
