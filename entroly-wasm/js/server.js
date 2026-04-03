#!/usr/bin/env node
// Entroly MCP Server — JS port of server.py
// Pure-wasm MCP server — no Python dependency.
//
// Architecture: MCP Client → JSON-RPC (stdio) → Node.js → Wasm Engine → Results

const { WasmEntrolyEngine } = require('../pkg/entroly_wasm');
const { EntrolyConfig } = require('./config');
const { CheckpointManager, persistIndex, loadIndex } = require('./checkpoint');
const { autoIndex, startIncrementalWatcher } = require('./auto_index');
const { startAutotuneDaemon, FeedbackJournal, TaskProfileOptimizer } = require('./autotune');
const path = require('path');
const fs = require('fs');

// ── MCP Protocol Implementation (stdio JSON-RPC 2.0) ──

class EntrolyMCPServer {
  constructor(config) {
    this.config = config || new EntrolyConfig();
    this.engine = new WasmEntrolyEngine();
    this.turnCounter = 0;

    // Feedback journal — persists episodes for cross-session autotune
    this.feedbackJournal = new FeedbackJournal(this.config.checkpointDir);
    // Task-conditioned weight profiles (novel: per-task-type optimization)
    this.taskProfiles = new TaskProfileOptimizer(this.feedbackJournal);
    this.taskProfiles.optimizeAll(); // warm from existing journal
    // Track last optimization context for feedback attribution
    this._lastOptCtx = null;

    // Checkpoint manager
    this.checkpointMgr = new CheckpointManager(
      this.config.checkpointDir, this.config.autoCheckpointInterval
    );

    // Index path for persistent repo-level indexing
    this.indexPath = path.join(this.config.checkpointDir, 'index.json.gz');

    // Try loading persistent index
    if (loadIndex(this.engine, this.indexPath)) {
      const n = this.engine.fragment_count();
      this._log(`Loaded persistent index: ${n} fragments`);
    }

    this._buffer = '';
    this._initialized = false;
  }

  _log(msg) { process.stderr.write(`${new Date().toISOString()} [entroly] ${msg}\n`); }

  // ── MCP Tool Definitions ──
  get tools() {
    return [
      {
        name: 'remember_fragment',
        description: 'Store a context fragment with automatic dedup and entropy scoring.',
        inputSchema: {
          type: 'object',
          properties: {
            content: { type: 'string', description: 'Text content to store' },
            source: { type: 'string', description: 'Origin label (e.g. file:utils.py)', default: '' },
            token_count: { type: 'integer', description: 'Token count (auto if 0)', default: 0 },
            is_pinned: { type: 'boolean', description: 'Always include in optimized context', default: false },
          },
          required: ['content'],
        },
      },
      {
        name: 'optimize_context',
        description: 'Select the optimal context subset for a token budget. Uses IOS + PRISM RL + Channel Coding.',
        inputSchema: {
          type: 'object',
          properties: {
            token_budget: { type: 'integer', description: 'Max tokens (default 128K)', default: 128000 },
            query: { type: 'string', description: 'Current task/query for semantic scoring', default: '' },
          },
        },
      },
      {
        name: 'recall_relevant',
        description: 'Semantic recall of most relevant stored fragments.',
        inputSchema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            top_k: { type: 'integer', description: 'Number of results', default: 5 },
          },
          required: ['query'],
        },
      },
      {
        name: 'record_outcome',
        description: 'Record whether selected fragments led to success/failure (RL feedback).',
        inputSchema: {
          type: 'object',
          properties: {
            fragment_ids: { type: 'string', description: 'Comma-separated fragment IDs' },
            success: { type: 'boolean', description: 'True if output was good', default: true },
          },
          required: ['fragment_ids'],
        },
      },
      {
        name: 'explain_context',
        description: 'Explain why each fragment was included/excluded in last optimization.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'get_stats',
        description: 'Get comprehensive session statistics.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'checkpoint_state',
        description: 'Save current state for crash recovery.',
        inputSchema: {
          type: 'object',
          properties: {
            task_description: { type: 'string', default: '' },
          },
        },
      },
      {
        name: 'resume_state',
        description: 'Resume from the latest checkpoint.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'analyze_codebase_health',
        description: 'Analyze codebase health (grade A-F).',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'security_report',
        description: 'Session-wide security audit across all ingested fragments.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'entroly_dashboard',
        description: 'Show live value metrics: money saved, performance, bloat prevention, quality.',
        inputSchema: { type: 'object', properties: {} },
      },
      {
        name: 'scan_for_vulnerabilities',
        description: 'Scan code for security vulnerabilities (SAST).',
        inputSchema: {
          type: 'object',
          properties: {
            content: { type: 'string', description: 'Source code to scan' },
            source: { type: 'string', description: 'File path', default: 'unknown' },
          },
          required: ['content'],
        },
      },
    ];
  }

  // ── Tool Handlers ──
  handleTool(name, args) {
    switch (name) {
      case 'remember_fragment':
        return this.engine.ingest(
          args.content, args.source || '', args.token_count || 0, args.is_pinned || false
        );

      case 'optimize_context': {
        this.turnCounter++;
        this.engine.advance_turn();
        const budget = args.token_budget || this.config.defaultTokenBudget;
        const query = args.query || '';

        // NOVEL: Apply task-conditioned weights before optimization
        const profile = this.taskProfiles.applyToEngine(this.engine, query);

        const result = this.engine.optimize(budget, query);
        result._taskProfile = { taskType: profile.taskType, confidence: profile.confidence };

        // Capture optimization context for feedback attribution
        const state = this.engine.export_state();
        this._lastOptCtx = {
          weights: { w_r: state.w_recency, w_f: state.w_frequency, w_s: state.w_semantic, w_e: state.w_entropy },
          selectedSources: (result.selected || []).map(s => s.source).filter(Boolean),
          selectedCount: result.selected_count || 0,
          tokenBudget: budget,
          query,
          turn: this.turnCounter,
        };

        // Auto-checkpoint
        if (this.checkpointMgr.shouldAutoCheckpoint()) {
          try {
            persistIndex(this.engine, this.indexPath);
            this.checkpointMgr.save({
              engine_state: state,
              turn: this.turnCounter,
            });
          } catch {}
        }
        return result;
      }

      case 'recall_relevant':
        return this.engine.recall(args.query, args.top_k || 5);

      case 'record_outcome': {
        const ids = (args.fragment_ids || '').split(',').map(s => s.trim()).filter(Boolean);
        const idsJson = JSON.stringify(ids);
        const success = args.success !== false;
        if (success) this.engine.record_success(idsJson);
        else this.engine.record_failure(idsJson);

        // Log feedback episode to journal for cross-session autotune
        if (this._lastOptCtx) {
          this.feedbackJournal.log({
            ...this._lastOptCtx,
            reward: success ? 1.0 : -1.0,
          });
        }

        return { status: 'recorded', fragment_ids: ids, outcome: success ? 'success' : 'failure' };
      }

      case 'explain_context':
        return this.engine.explain_selection();

      case 'get_stats': {
        const stats = this.engine.stats();
        stats.checkpoint = this.checkpointMgr.stats();
        return stats;
      }

      case 'checkpoint_state': {
        try {
          persistIndex(this.engine, this.indexPath);
          const p = this.checkpointMgr.save({
            engine_state: this.engine.export_state(),
            turn: this.turnCounter,
            task: args.task_description || '',
          });
          return { status: 'checkpoint_saved', path: p };
        } catch (e) {
          return { status: 'error', error: e.message };
        }
      }

      case 'resume_state': {
        const ckpt = this.checkpointMgr.loadLatest();
        if (!ckpt) return { status: 'no_checkpoint_found' };
        if (ckpt.engine_state) {
          this.engine.import_state(JSON.stringify(ckpt.engine_state));
        }
        return { status: 'resumed', checkpoint_id: ckpt.checkpoint_id, turn: ckpt.turn || 0 };
      }

      case 'analyze_codebase_health':
        return this.engine.analyze_health();

      case 'security_report':
        return this.engine.security_report();

      case 'scan_for_vulnerabilities':
        return this.engine.scan_fragment(args.source || 'unknown');

      case 'entroly_dashboard': {
        const stats = this.engine.stats();
        const explanation = this.engine.explain_selection();
        return { stats, explanation, turn: this.turnCounter };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  // ── JSON-RPC stdio transport ──
  async run() {
    this._log(`Starting Entroly MCP server v${this.config.serverVersion} (Wasm engine)`);

    // Auto-index on startup
    try {
      const result = autoIndex(this.engine);
      if (result.status === 'indexed') {
        this._log(`Auto-indexed ${result.files_indexed} files (${result.total_tokens.toLocaleString()} tokens) in ${result.duration_s}s`);
      }
      startIncrementalWatcher(this.engine);
    } catch (e) {
      this._log(`Auto-index failed (non-fatal): ${e.message}`);
    }

    // Start autotune daemon — reads tuning_config.json, hot-reloads weights
    try {
      const tid = startAutotuneDaemon(this.engine);
      if (tid) this._log('Autotune daemon started (hot-reload every 30s)');
    } catch (e) {
      this._log(`Autotune: failed to start daemon: ${e.message}`);
    }

    // Graceful shutdown
    process.on('SIGTERM', () => {
      this._log('Shutdown — persisting state...');
      try { persistIndex(this.engine, this.indexPath); } catch {}
      process.exit(0);
    });

    // Read JSONRPC from stdin
    process.stdin.setEncoding('utf-8');
    process.stdin.on('data', (chunk) => {
      this._buffer += chunk;
      this._processBuffer();
    });
    process.stdin.on('end', () => {
      this._log('stdin closed — shutting down');
      try { persistIndex(this.engine, this.indexPath); } catch {}
    });
  }

  _processBuffer() {
    // MCP uses Content-Length framed JSON-RPC
    while (true) {
      const headerEnd = this._buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) return;

      const header = this._buffer.slice(0, headerEnd);
      const match = header.match(/Content-Length:\s*(\d+)/i);
      if (!match) {
        // Try newline-delimited JSON (simpler transport)
        const nlIdx = this._buffer.indexOf('\n');
        if (nlIdx === -1) return;
        const line = this._buffer.slice(0, nlIdx).trim();
        this._buffer = this._buffer.slice(nlIdx + 1);
        if (line) {
          try {
            const msg = JSON.parse(line);
            this._handleMessage(msg);
          } catch {}
        }
        continue;
      }

      const contentLength = parseInt(match[1], 10);
      const bodyStart = headerEnd + 4;
      if (this._buffer.length < bodyStart + contentLength) return;

      const body = this._buffer.slice(bodyStart, bodyStart + contentLength);
      this._buffer = this._buffer.slice(bodyStart + contentLength);

      try {
        const msg = JSON.parse(body);
        this._handleMessage(msg);
      } catch (e) {
        this._log(`JSON parse error: ${e.message}`);
      }
    }
  }

  _handleMessage(msg) {
    if (msg.method === 'initialize') {
      this._initialized = true;
      this._respond(msg.id, {
        protocolVersion: '2024-11-05',
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: this.config.serverName, version: this.config.serverVersion },
      });
    } else if (msg.method === 'notifications/initialized') {
      // No response needed for notifications
    } else if (msg.method === 'tools/list') {
      this._respond(msg.id, { tools: this.tools });
    } else if (msg.method === 'tools/call') {
      const { name, arguments: toolArgs } = msg.params;
      try {
        const result = this.handleTool(name, toolArgs || {});
        const text = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
        this._respond(msg.id, {
          content: [{ type: 'text', text }], isError: false,
        });
      } catch (e) {
        this._respond(msg.id, {
          content: [{ type: 'text', text: JSON.stringify({ error: e.message }) }], isError: true,
        });
      }
    } else if (msg.method === 'ping') {
      this._respond(msg.id, {});
    } else if (msg.id !== undefined) {
      // Unknown method with ID — respond with error
      this._respondError(msg.id, -32601, `Method not found: ${msg.method}`);
    }
  }

  _respond(id, result) {
    if (id === undefined) return;
    const response = JSON.stringify({ jsonrpc: '2.0', id, result });
    const header = `Content-Length: ${Buffer.byteLength(response)}\r\n\r\n`;
    process.stdout.write(header + response);
  }

  _respondError(id, code, message) {
    const response = JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } });
    const header = `Content-Length: ${Buffer.byteLength(response)}\r\n\r\n`;
    process.stdout.write(header + response);
  }
}

// ── Entry Point ──
if (require.main === module) {
  const server = new EntrolyMCPServer();
  server.run();
}

module.exports = { EntrolyMCPServer };
