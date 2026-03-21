"""
Entroly Live Dashboard — Real-time AI value metrics at localhost:9378
=====================================================================

Shows developers exactly what Entroly's Rust engine is doing for them,
pulling REAL data from all engine subsystems:

  Engine Stats:       tokens saved, cost saved, dedup hits, turn count
  PRISM RL Weights:   learned scoring weights (recency/frequency/semantic/entropy)
  Health Analysis:    code health grade A–F, clones, dead symbols, god files
  SAST Security:      vulnerability findings with CWE categories
  Knapsack Decisions: which fragments were included/excluded and why
  Dep Graph:          symbol definitions, edges, coupling stats

Starts alongside the proxy and auto-refreshes every 3 seconds.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

logger = logging.getLogger("entroly.dashboard")

# ── Engine reference (set by start_dashboard) ─────────────────────────────────
_engine: Optional[Any] = None
_lock = threading.Lock()

# Per-request tracking (populated by proxy integration)
_request_log: list[dict] = []
_MAX_LOG = 50


def record_request(entry: dict):
    """Record a proxy request's metrics (called from proxy.py)."""
    with _lock:
        _request_log.append(entry)
        if len(_request_log) > _MAX_LOG:
            del _request_log[: len(_request_log) - _MAX_LOG]


def _safe_json(obj: Any) -> Any:
    """Recursively convert to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return 0.0
        return round(obj, 6)
    return obj


def _get_full_snapshot() -> dict:
    """Pull ALL real data from the engine subsystems."""
    snap: dict[str, Any] = {
        "ts": time.time(),
        "engine_available": _engine is not None,
    }

    if _engine is None:
        return snap

    try:
        # 1. Core stats — tokens saved, cost, dedup, turns
        stats = _engine.stats() if hasattr(_engine, "stats") else {}
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            stats = _engine._rust.stats()
            stats = dict(stats)
            for k, v in stats.items():
                if hasattr(v, "items"):
                    stats[k] = dict(v)
        snap["stats"] = _safe_json(stats)
    except Exception as e:
        snap["stats"] = {"error": str(e)}

    try:
        # 2. PRISM RL weights — the learned scoring weights
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            rust = _engine._rust
            snap["prism_weights"] = {
                "recency": round(getattr(rust, "w_recency", 0.3), 4),
                "frequency": round(getattr(rust, "w_frequency", 0.25), 4),
                "semantic": round(getattr(rust, "w_semantic", 0.25), 4),
                "entropy": round(getattr(rust, "w_entropy", 0.2), 4),
            }
    except Exception:
        snap["prism_weights"] = None

    try:
        # 3. Health analysis — code health grade
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            health_json = _engine._rust.analyze_health()
            snap["health"] = _safe_json(json.loads(health_json))
    except Exception:
        snap["health"] = None

    try:
        # 4. SAST security report
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            sec_json = _engine._rust.security_report()
            snap["security"] = _safe_json(json.loads(sec_json))
    except Exception:
        snap["security"] = None

    try:
        # 5. Knapsack explainability — last optimization decisions
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            explain = _engine._rust.explain_selection()
            snap["explain"] = _safe_json(dict(explain))
    except Exception:
        snap["explain"] = None

    try:
        # 6. Dependency graph stats
        if hasattr(_engine, "_use_rust") and _engine._use_rust:
            dg = _engine._rust.dep_graph_stats()
            snap["dep_graph"] = _safe_json(dict(dg))
    except Exception:
        snap["dep_graph"] = None

    # 7. Recent proxy requests
    with _lock:
        snap["recent_requests"] = list(_request_log)

    return snap


# ── HTML Dashboard ────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Entroly — Intelligence Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #050508; --bg2: #0a0b10; --card: rgba(14,17,24,0.85);
  --glass: rgba(255,255,255,0.03); --glass2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.06); --border2: rgba(255,255,255,0.12);
  --text: #e8ecf4; --dim: #6b7280; --dim2: #3b4252;
  --emerald: #34d399; --emerald-glow: rgba(52,211,153,0.15);
  --blue: #60a5fa; --blue-glow: rgba(96,165,250,0.12);
  --violet: #a78bfa; --violet-glow: rgba(167,139,250,0.12);
  --amber: #fbbf24; --amber-glow: rgba(251,191,36,0.10);
  --rose: #fb7185; --rose-glow: rgba(251,113,133,0.10);
  --cyan: #22d3ee; --cyan-glow: rgba(34,211,238,0.10);
  --grad1: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  --grad2: linear-gradient(135deg, #34d399 0%, #06b6d4 100%);
  --grad3: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;
  background:radial-gradient(circle at 30% 20%,rgba(102,126,234,0.04),transparent 50%),
  radial-gradient(circle at 70% 80%,rgba(118,75,162,0.03),transparent 50%);z-index:0;pointer-events:none;}

/* Top Bar */
.topbar{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
  padding:14px 32px;background:rgba(5,5,8,0.8);backdrop-filter:blur(20px);border-bottom:1px solid var(--border);}
.brand{display:flex;align-items:center;gap:14px;}
.brand h1{font-size:24px;font-weight:900;letter-spacing:-0.5px;
  background:var(--grad1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.brand .tag{font-size:11px;padding:3px 10px;border-radius:20px;background:var(--emerald-glow);
  color:var(--emerald);font-weight:600;letter-spacing:0.5px;}
.live{display:flex;align-items:center;gap:8px;color:var(--emerald);font-size:12px;font-weight:500;}
.live .dot{width:7px;height:7px;border-radius:50%;background:var(--emerald);
  box-shadow:0 0 12px var(--emerald);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 12px var(--emerald);}50%{opacity:0.4;box-shadow:0 0 4px var(--emerald);}}

/* Layout */
.main{position:relative;z-index:1;padding:24px 32px;max-width:1440px;margin:0 auto;}

/* Savings Hero */
.savings-hero{display:flex;align-items:center;gap:32px;padding:32px 40px;margin-bottom:24px;
  background:var(--card);border:1px solid var(--border);border-radius:20px;position:relative;overflow:hidden;}
.savings-hero::before{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(52,211,153,0.05),rgba(96,165,250,0.03),transparent);pointer-events:none;}
.savings-main{flex:1;}
.savings-label{font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim);margin-bottom:8px;}
.savings-value{font-size:64px;font-weight:900;letter-spacing:-3px;font-feature-settings:'tnum';
  background:var(--grad2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  filter:drop-shadow(0 0 30px rgba(52,211,153,0.3));transition:all 0.6s cubic-bezier(0.16,1,0.3,1);}
.savings-sub{font-size:13px;color:var(--dim);margin-top:4px;}
.savings-metrics{display:flex;gap:40px;}
.smetric{text-align:center;}
.smetric .sv{font-size:28px;font-weight:800;letter-spacing:-1px;font-feature-settings:'tnum';}
.smetric .sl{font-size:11px;color:var(--dim);margin-top:4px;text-transform:uppercase;letter-spacing:1px;}
.sv-tokens{color:var(--blue);}
.sv-dedup{color:var(--amber);}
.sv-turns{color:var(--violet);}
.sv-frag{color:var(--cyan);}

/* Grid */
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px;}

/* Panel */
.panel{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;
  backdrop-filter:blur(10px);transition:border-color 0.3s,box-shadow 0.3s;}
.panel:hover{border-color:var(--border2);box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.ph{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border);}
.ph h2{font-size:14px;font-weight:700;}
.badge{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;}
.b-green{background:var(--emerald-glow);color:var(--emerald);}
.b-blue{background:var(--blue-glow);color:var(--blue);}
.b-violet{background:var(--violet-glow);color:var(--violet);}
.b-amber{background:var(--amber-glow);color:var(--amber);}
.b-rose{background:var(--rose-glow);color:var(--rose);}
.b-cyan{background:var(--cyan-glow);color:var(--cyan);}
.pb{padding:20px;}

/* PRISM Radar */
.radar-wrap{display:flex;align-items:center;justify-content:center;padding:16px 0;}
.radar-canvas{width:200px;height:200px;}
.radar-legend{list-style:none;margin-left:24px;}
.radar-legend li{display:flex;align-items:center;gap:8px;padding:6px 0;font-size:13px;color:var(--dim);}
.radar-legend .rdot{width:8px;height:8px;border-radius:50%;}
.radar-legend .rval{font-weight:700;color:var(--text);font-feature-settings:'tnum';min-width:36px;}

/* Health Ring */
.health-ring-wrap{display:flex;align-items:center;justify-content:center;gap:28px;padding:20px 0;}
.health-ring{position:relative;width:120px;height:120px;}
.health-ring canvas{width:100%;height:100%;}
.health-ring .grade{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  font-size:42px;font-weight:900;}
.health-stats{list-style:none;}
.health-stats li{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px;color:var(--dim);}
.health-stats .hv{font-weight:700;color:var(--text);min-width:20px;text-align:right;}

/* Tables */
table{width:100%;border-collapse:collapse;}
th{font-size:10px;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim2);padding:10px 14px;
  text-align:left;background:rgba(255,255,255,0.02);font-weight:600;}
td{padding:8px 14px;border-top:1px solid var(--border);font-size:13px;}
td.mono{font-family:'JetBrains Mono',monospace;font-size:12px;}
tr:hover td{background:rgba(255,255,255,0.015);}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;}
.t-green{background:var(--emerald-glow);color:var(--emerald);}
.t-rose{background:var(--rose-glow);color:var(--rose);}
.t-amber{background:var(--amber-glow);color:var(--amber);}
.t-violet{background:var(--violet-glow);color:var(--violet);}

/* Empty */
.empty{text-align:center;padding:32px;color:var(--dim2);font-size:13px;}

/* Security Shield */
.shield-ok{text-align:center;padding:24px;}
.shield-icon{font-size:48px;margin-bottom:8px;filter:drop-shadow(0 0 20px rgba(52,211,153,0.4));}
.shield-text{color:var(--emerald);font-weight:700;font-size:15px;}
.shield-sub{color:var(--dim);font-size:12px;margin-top:4px;}

/* Request sparkline */
.sparkline{display:flex;align-items:flex-end;gap:2px;height:40px;margin-top:8px;}
.sparkline .bar{flex:1;background:var(--grad2);border-radius:2px 2px 0 0;min-width:3px;
  transition:height 0.4s cubic-bezier(0.16,1,0.3,1);opacity:0.7;}
.sparkline .bar:hover{opacity:1;}

/* Responsive */
@media(max-width:1100px){.grid3{grid-template-columns:1fr 1fr;}.savings-hero{flex-direction:column;gap:20px;}.savings-metrics{flex-wrap:wrap;}}
@media(max-width:768px){.grid2,.grid3{grid-template-columns:1fr;}.main{padding:16px;}.savings-value{font-size:48px;}}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">
    <h1>⚡ Entroly</h1>
    <span class="tag">INTELLIGENCE DASHBOARD</span>
  </div>
  <div class="live"><div class="dot"></div>Live · 3s refresh</div>
</div>

<div class="main">
  <!-- Savings Hero -->
  <div class="savings-hero" id="hero"></div>

  <!-- PRISM + Health -->
  <div class="grid2">
    <div class="panel">
      <div class="ph"><h2>🧠 PRISM Intelligence</h2><span class="badge b-violet">RL-Learned</span></div>
      <div class="pb" id="prism"></div>
    </div>
    <div class="panel">
      <div class="ph"><h2>🏥 Code Health</h2><span id="hb" class="badge b-green">—</span></div>
      <div class="pb" id="health"></div>
    </div>
  </div>

  <!-- Security + Dep Graph + Knapsack -->
  <div class="grid3">
    <div class="panel">
      <div class="ph"><h2>🛡️ Security</h2><span id="sb" class="badge b-green">Clean</span></div>
      <div class="pb" id="security"></div>
    </div>
    <div class="panel">
      <div class="ph"><h2>🕸️ Dep Graph</h2><span id="db" class="badge b-cyan">—</span></div>
      <div class="pb" id="depgraph"></div>
    </div>
    <div class="panel">
      <div class="ph"><h2>🎯 Knapsack</h2><span id="kb" class="badge b-violet">—</span></div>
      <div class="pb" id="knapsack" style="max-height:320px;overflow-y:auto;"></div>
    </div>
  </div>

  <!-- Requests -->
  <div class="panel" style="margin-bottom:28px;">
    <div class="ph"><h2>📡 Request Flow</h2><span id="rb" class="badge b-cyan">—</span></div>
    <div id="sparkarea" style="padding:12px 20px 0;"></div>
    <div style="overflow-x:auto;">
      <table><thead><tr>
        <th>Time</th><th>Model</th><th>Tokens In</th><th>Saved</th><th>Dedup</th><th>SAST</th><th>Query</th>
      </tr></thead><tbody id="reqs"></tbody></table>
    </div>
  </div>
</div>

<script>
const fmt=n=>{if(n==null)return'—';return n>=1e6?(n/1e6).toFixed(1)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n)};
const money=n=>'$'+(n||0).toFixed(2);
const pct=n=>Math.round((n||0)*100)+'%';
const ago=ts=>{const s=Math.floor(Date.now()/1000-ts);return s<60?s+'s ago':s<3600?Math.floor(s/60)+'m ago':Math.floor(s/3600)+'h ago';};

let prevCost=0;
function renderHero(d){
  const s=d.stats||{},sv=s.savings||{},ss=s.session||{},dd=s.dedup||{};
  const cost=sv.estimated_cost_saved_usd||0;
  const tokens=sv.total_tokens_saved||0;
  const dups=sv.total_duplicates_caught||dd.duplicates_detected||0;
  const turns=ss.current_turn||0;
  const frags=ss.total_fragments||0;
  const ent=ss.avg_entropy||ss.avg_entropy_score||0;

  document.getElementById('hero').innerHTML=`
    <div class="savings-main">
      <div class="savings-label">Total Value Delivered</div>
      <div class="savings-value">${money(cost)}</div>
      <div class="savings-sub">${fmt(sv.total_optimizations||0)} optimizations · avg entropy ${(ent||0).toFixed(3)}</div>
    </div>
    <div class="savings-metrics">
      <div class="smetric"><div class="sv sv-tokens">${fmt(tokens)}</div><div class="sl">Tokens Saved</div></div>
      <div class="smetric"><div class="sv sv-dedup">${dups}</div><div class="sl">Dedup Hits</div></div>
      <div class="smetric"><div class="sv sv-frag">${fmt(frags)}</div><div class="sl">Fragments</div></div>
      <div class="smetric"><div class="sv sv-turns">${turns}</div><div class="sl">Turns</div></div>
    </div>`;
  prevCost=cost;
}

function drawRadar(ctx,w,vals,colors){
  const cx=w/2,cy=w/2,r=w/2-20,n=vals.length;
  ctx.clearRect(0,0,w,w);
  // Grid rings
  for(let i=1;i<=4;i++){
    ctx.beginPath();
    for(let j=0;j<=n;j++){
      const a=Math.PI*2*j/n-Math.PI/2;
      const rr=r*i/4;
      j===0?ctx.moveTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a)):ctx.lineTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a));
    }
    ctx.strokeStyle='rgba(255,255,255,0.06)';ctx.stroke();
  }
  // Data
  ctx.beginPath();
  vals.forEach((v,i)=>{
    const a=Math.PI*2*i/n-Math.PI/2;
    const rr=r*Math.min(v/0.5,1);
    i===0?ctx.moveTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a)):ctx.lineTo(cx+rr*Math.cos(a),cy+rr*Math.sin(a));
  });
  ctx.closePath();
  ctx.fillStyle='rgba(167,139,250,0.15)';ctx.fill();
  ctx.strokeStyle='rgba(167,139,250,0.8)';ctx.lineWidth=2;ctx.stroke();
  // Dots
  vals.forEach((v,i)=>{
    const a=Math.PI*2*i/n-Math.PI/2;
    const rr=r*Math.min(v/0.5,1);
    ctx.beginPath();ctx.arc(cx+rr*Math.cos(a),cy+rr*Math.sin(a),4,0,Math.PI*2);
    ctx.fillStyle=colors[i];ctx.fill();
    ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  });
}

function renderPrism(d){
  const w=d.prism_weights;
  if(!w){document.getElementById('prism').innerHTML='<div class="empty">Engine not initialized</div>';return;}
  const names=['Recency','Frequency','Semantic','Entropy'];
  const vals=[w.recency,w.frequency,w.semantic,w.entropy];
  const colors=['#667eea','#f5576c','#4facfe','#43e97b'];
  const el=document.getElementById('prism');
  el.innerHTML=`<div class="radar-wrap">
    <canvas class="radar-canvas" id="radarC" width="200" height="200"></canvas>
    <ul class="radar-legend">${names.map((n,i)=>`
      <li><span class="rdot" style="background:${colors[i]}"></span>${n}<span class="rval">${pct(vals[i])}</span></li>`).join('')}
      <li style="margin-top:8px;font-size:11px;color:var(--dim2);">Weights evolve via spectral RL</li>
    </ul>
  </div>`;
  const c=document.getElementById('radarC');
  if(c)drawRadar(c.getContext('2d'),200,vals,colors);
}

function renderHealth(d){
  const h=d.health,el=document.getElementById('health'),b=document.getElementById('hb');
  if(!h||h.error){el.innerHTML='<div class="empty">Ingest code to see health</div>';return;}
  const g=h.health_grade||'?',sc=h.code_health_score||0;
  const gc={'A':'var(--emerald)','B':'var(--blue)','C':'var(--amber)','D':'#e3872d','F':'var(--rose)'}[g]||'var(--dim)';
  b.textContent=g+' · '+sc+'/100';
  b.className='badge '+(g<='B'?'b-green':g==='C'?'b-amber':'b-rose');
  el.innerHTML=`<div class="health-ring-wrap">
    <div class="health-ring">
      <canvas id="hring" width="120" height="120"></canvas>
      <div class="grade" style="color:${gc}">${g}</div>
    </div>
    <ul class="health-stats">
      <li><span class="hv">${(h.clone_pairs||[]).length}</span>clone pairs</li>
      <li><span class="hv">${(h.dead_symbols||[]).length}</span>dead symbols</li>
      <li><span class="hv">${(h.god_files||[]).length}</span>god files</li>
      <li><span class="hv">${(h.arch_violations||[]).length}</span>arch violations</li>
      <li><span class="hv">${(h.naming_issues||[]).length}</span>naming issues</li>
    </ul>
  </div>${h.top_recommendation?'<div style="padding:10px;background:rgba(251,191,36,0.06);border-radius:10px;font-size:12px;color:var(--amber);">💡 '+h.top_recommendation+'</div>':''}`;
  // Draw ring
  const c=document.getElementById('hring');
  if(c){const ctx=c.getContext('2d'),cx=60,cy=60,r=50,pct2=sc/100;
    ctx.beginPath();ctx.arc(cx,cy,r,0,Math.PI*2);ctx.strokeStyle='rgba(255,255,255,0.05)';ctx.lineWidth=8;ctx.stroke();
    ctx.beginPath();ctx.arc(cx,cy,r,-Math.PI/2,-Math.PI/2+Math.PI*2*pct2);ctx.strokeStyle=gc;ctx.lineWidth=8;ctx.lineCap='round';ctx.stroke();}
}

function renderSecurity(d){
  const s=d.security,el=document.getElementById('security'),b=document.getElementById('sb');
  if(!s||s.error){el.innerHTML='<div class="empty">No scan yet</div>';return;}
  const tot=(s.critical_total||0)+(s.high_total||0);
  if(tot===0){b.textContent='✓ Clean';b.className='badge b-green';
    el.innerHTML=`<div class="shield-ok"><div class="shield-icon">🛡️</div><div class="shield-text">No vulnerabilities</div><div class="shield-sub">${s.fragments_scanned||0} fragments scanned</div></div>`;return;}
  b.textContent=tot+' findings';b.className='badge '+(s.critical_total>0?'b-rose':'b-amber');
  const cats=s.findings_by_category||{};
  el.innerHTML=`<div style="display:flex;gap:16px;margin-bottom:12px;text-align:center;">
    <div style="flex:1;"><div style="font-size:24px;font-weight:800;color:var(--rose);">${s.critical_total||0}</div><div style="font-size:10px;color:var(--dim);">CRITICAL</div></div>
    <div style="flex:1;"><div style="font-size:24px;font-weight:800;color:var(--amber);">${s.high_total||0}</div><div style="font-size:10px;color:var(--dim);">HIGH</div></div>
  </div><ul style="list-style:none;">${Object.entries(cats).map(([k,v])=>`<li style="display:flex;justify-content:space-between;padding:3px 0;"><span style="font-size:12px;color:var(--dim);">${k}</span><span class="tag t-rose">${v}</span></li>`).join('')}</ul>`;
}

function renderDepGraph(d){
  const dg=d.dep_graph,el=document.getElementById('depgraph'),b=document.getElementById('db');
  if(!dg){el.innerHTML='<div class="empty">Run optimize first</div>';return;}
  const sym=dg.total_symbols||dg.symbol_count||0,edg=dg.total_edges||dg.edge_count||0;
  b.textContent=sym+' symbols';
  el.innerHTML=`<div style="display:flex;gap:20px;justify-content:center;padding:16px 0;">
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:var(--cyan);">${sym}</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">SYMBOLS</div></div>
    <div style="text-align:center;"><div style="font-size:32px;font-weight:800;color:var(--blue);">${edg}</div><div style="font-size:10px;color:var(--dim);margin-top:4px;">EDGES</div></div>
  </div><div style="font-size:11px;color:var(--dim2);text-align:center;">Cross-file dependency tracking</div>`;
}

function renderKnapsack(d){
  const ex=d.explain,el=document.getElementById('knapsack'),b=document.getElementById('kb');
  if(!ex||ex.error){el.innerHTML='<div class="empty">Run optimize first</div>';return;}
  const inc=ex.included||[],exc=ex.excluded||[];
  b.textContent=inc.length+' selected · '+pct(ex.sufficiency)+' suff.';
  let rows=inc.slice(0,6).map(f=>{const s=f.scores||{};
    return`<tr><td class="mono" style="color:var(--emerald);">✓ ${(f.source||f.id||'').split('/').pop()}</td><td class="mono">${pct(s.composite)}</td><td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,35)}</td></tr>`;}).join('');
  rows+=exc.slice(0,3).map(f=>{const s=f.scores||{};
    return`<tr style="opacity:0.4;"><td class="mono" style="color:var(--rose);">✗ ${(f.source||f.id||'').split('/').pop()}</td><td class="mono">${pct(s.composite)}</td><td style="font-size:11px;color:var(--dim);">${(f.reason||'').slice(0,35)}</td></tr>`;}).join('');
  el.innerHTML=`<table><thead><tr><th>Fragment</th><th>Score</th><th>Reason</th></tr></thead><tbody>${rows||'<tr><td colspan="3" class="empty">No data</td></tr>'}</tbody></table>`;
}

let sparkData=[];
function renderRequests(d){
  const reqs=d.recent_requests||[],tbody=document.getElementById('reqs'),b=document.getElementById('rb');
  b.textContent=reqs.length+' recent';
  // Sparkline
  if(reqs.length>0){
    reqs.forEach(r=>{if(sparkData.length>=30)sparkData.shift();sparkData.push(r.tokens_saved||0);});
    const mx=Math.max(...sparkData,1);
    document.getElementById('sparkarea').innerHTML=`<div class="sparkline">${sparkData.map(v=>`<div class="bar" style="height:${Math.max(2,v/mx*40)}px;"></div>`).join('')}</div>`;
  }
  if(reqs.length===0){tbody.innerHTML='<tr><td colspan="7" class="empty">No requests yet — proxy on :9377</td></tr>';return;}
  tbody.innerHTML=reqs.slice().reverse().slice(0,15).map(r=>`<tr>
    <td>${ago(r.time||0)}</td><td>${r.model||'—'}</td><td class="mono">${fmt(r.tokens_in||0)}</td>
    <td><span class="tag t-green">−${fmt(r.tokens_saved||0)}</span></td>
    <td>${(r.dedup_hits||0)>0?'<span class="tag t-amber">'+r.dedup_hits+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td>${(r.sast_findings||0)>0?'<span class="tag t-rose">'+r.sast_findings+'</span>':'<span style="color:var(--dim2)">0</span>'}</td>
    <td style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--dim);">${r.query||'—'}</td>
  </tr>`).join('');
}

async function refresh(){
  try{const r=await fetch('/api/metrics');const d=await r.json();
    renderHero(d);renderPrism(d);renderHealth(d);renderSecurity(d);renderDepGraph(d);renderKnapsack(d);renderRequests(d);
  }catch(e){console.error('Refresh:',e);}
}
refresh();setInterval(refresh,3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress access logs

    def do_GET(self):
        if self.path == "/" or self.path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            snap = _get_full_snapshot()
            self.wfile.write(json.dumps(snap, default=str).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()


def start_dashboard(engine: Any = None, port: int = 9378, daemon: bool = True):
    """
    Start the dashboard HTTP server in a background thread.

    Args:
        engine: The EntrolyEngine instance to pull real data from.
        port: Port to serve on (default: 9378).
        daemon: Run as daemon thread (dies with main process).

    Returns:
        The HTTPServer instance.
    """
    global _engine
    _engine = engine

    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=daemon)
    thread.start()
    logger.info(f"Dashboard live at http://localhost:{port}")
    return server
