with open("README.md", "r") as f:
    original_content = f.read()

# Professional introduction with video
professional_header = """# Entroly

[![PyPI](https://img.shields.io/pypi/v/entroly)](https://pypi.org/project/entroly/)
[![CI](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/entroly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/core-Rust%20%2B%20PyO3-orange)](entroly-core/)

**Information-theoretic context optimization for AI coding agents.**

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

"""

# Find the point after the existing badges/titles to insert, or just replace the very top section
# The original file starts with # Entroly and some badges. 
# We'll just replace the first 10 lines or so.

lines = original_content.splitlines()
# Keep everything from the first "## Zero-Friction Setup" or similar section down.
# Let's see where the original content really starts getting unique.
start_index = 0
for i, line in enumerate(lines):
    if "## Zero-Friction Setup" in line:
        start_index = i
        break

if start_index == 0:
    # Fallback to after the first few lines
    start_index = 15

merged_content = professional_header + "\n".join(lines[start_index:])

with open("README.md", "w") as f:
    f.write(merged_content)
