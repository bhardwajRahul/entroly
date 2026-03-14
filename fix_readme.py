import re

with open("README.md.restored", "r") as f:
    text = f.read()

# Make the introductory text more professional and add the video
# Find "**Information-theoretic context optimization for AI coding agents.**"
start_str = "**Information-theoretic context optimization for AI coding agents.**"
idx = text.find(start_str)

top_replacement = """**Information-theoretic context optimization for AI coding agents.**

> **The Problem:** AI coding tools commonly manage context using FIFO (First-In, First-Out) truncation. When diagnosing an issue such as a SQL injection, the context window may become saturated with irrelevant files (e.g., CSS, READMEs, Docker configurations). This forces the truncation of critical components, leading to degraded LLM responses, wasted API credits, and necessary re-prompting.
> 
> **The Solution:** Entroly applies mathematical optimization to select the optimal subset of context. It ensures the LLM receives the most relevant code structure rather than simply the most recent additions.

<div align="center">
  <br/>
  <h2>Watch Entroly in Action (Live Engine Metrics)</h2>
  
  https://github.com/juyterman1000/entroly/raw/main/entroly_demo.mp4

  <p><i>The demonstration above illustrates the 100% Rust <code>entroly_core</code> engine executing mathematically optimal context selection in under a millisecond.</i></p>
</div>

---

## The Value Proposition

When deploying Entroly as your agent's MCP context server:

- **100% Signal, 0% Noise:** Automatically filters out irrelevant context such as markdown, stylesheets, and unrelated test files.
- **Dependency Resolution:** Automatically links related files via static import analysis. If `auth/db.py` is included, the engine recognizes the necessary dependency on `config/database.py`.
- **Deduplication:** Utilizes 64-bit SimHash to detect near-duplicates instantly, preventing the expenditure of tokens on redundant code fragments.
- **High Performance:** The 0/1 Knapsack optimization algorithm executes in under 0.5ms via the Rust core, ensuring no noticeable latency is added before the LLM request.

"""

text = text[:idx] + top_replacement + text[text.find("```bash\npip install entroly[native]\n```"):]

# Remove emojis globally for a professional tone
emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]|\u2600-\u27FF|\u25B6|\u23FAD', flags=re.UNICODE)
text = emoji_pattern.sub(r'', text)

with open("README.md", "w") as f:
    f.write(text)
