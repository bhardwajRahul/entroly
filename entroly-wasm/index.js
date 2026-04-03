// Entroly — Information-theoretic context optimization for JavaScript/TypeScript.
// 
// Usage:
//   const { EntrolyEngine, autoIndex, EntrolyConfig } = require('entroly-wasm');
//
//   const engine = new EntrolyEngine();
//   autoIndex(engine);   // auto-discover and ingest codebase
//   engine.optimize(8000, "fix the auth bug");
//
// Or as MCP server:
//   npx entroly-wasm serve

const { WasmEntrolyEngine } = require('./pkg/entroly_wasm');
const { EntrolyConfig } = require('./js/config');
const { autoIndex, startIncrementalWatcher, estimateTokens } = require('./js/auto_index');
const { CheckpointManager, persistIndex, loadIndex } = require('./js/checkpoint');
const { EntrolyMCPServer } = require('./js/server');
const { runAutotune, startAutotuneDaemon, TaskProfileOptimizer, FeedbackJournal } = require('./js/autotune');

module.exports = {
  // Core engine (wasm)
  EntrolyEngine: WasmEntrolyEngine,
  WasmEntrolyEngine,

  // Configuration
  EntrolyConfig,

  // Codebase scanning
  autoIndex,
  startIncrementalWatcher,
  estimateTokens,

  // State persistence
  CheckpointManager,
  persistIndex,
  loadIndex,

  // MCP Server
  EntrolyMCPServer,

  // Autotune + Task-Conditioned Profiles
  runAutotune,
  startAutotuneDaemon,
  TaskProfileOptimizer,
  FeedbackJournal,
};
