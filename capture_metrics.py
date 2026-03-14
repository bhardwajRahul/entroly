#!/usr/bin/env python3
"""Capture real Entroly engine metrics as JSON for the animated demo."""
import json, time
from entroly_core import EntrolyEngine

CODEBASE = {
    "auth/db.py": {
        "content": "import sqlite3\ndef get_user(user_id):\n    conn = sqlite3.connect('app.db')\n    cursor = conn.cursor()\n    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n    return cursor.fetchone()\ndef delete_user(user_id):\n    conn = sqlite3.connect('app.db')\n    cursor = conn.cursor()\n    cursor.execute(f'DELETE FROM users WHERE id = {user_id}')\n    conn.commit()",
        "tokens": 85, "relevant": True,
    },
    "auth/queries.py": {
        "content": "def parameterized_query(cursor, query, params):\n    cursor.execute(query, params)\n    return cursor.fetchall()\ndef build_where_clause(filters):\n    clauses = []\n    values = []\n    for key, val in filters.items():\n        clauses.append(f'{key} = ?')\n        values.append(val)\n    return ' AND '.join(clauses), values",
        "tokens": 72, "relevant": True,
    },
    "models/user.py": {
        "content": "from dataclasses import dataclass\nclass User:\n    id: int\n    email: str\n    password_hash: str\n    role: str = 'user'\n    def verify_password(self, password):\n        import hashlib\n        return hashlib.sha256(password.encode()).hexdigest() == self.password_hash",
        "tokens": 50, "relevant": True,
    },
    "config/database.py": {
        "content": "import os\nDB_HOST = os.environ.get('DB_HOST', 'localhost')\nDB_PORT = int(os.environ.get('DB_PORT', '5432'))\nDB_NAME = os.environ.get('DB_NAME', 'app_db')\nSQLALCHEMY_DATABASE_URI = f'postgresql://{DB_HOST}:{DB_PORT}/{DB_NAME}'\nSQLALCHEMY_POOL_SIZE = 20",
        "tokens": 40, "relevant": True,
    },
    "README.md": {
        "content": "# MyApp - Enterprise SaaS Platform\n## Quick Start\npip install -r requirements.txt\npython manage.py runserver\n## Architecture\nThis application uses Flask with PostgreSQL.",
        "tokens": 120, "relevant": False,
    },
    "static/styles.css": {
        "content": ":root { --primary: #2563eb; }\n.button { color: var(--primary); }\n.nav { display: flex; }\n.container { max-width: 1200px; margin: 0 auto; }\n.footer { background: #333; }\n.modal-overlay { position: fixed; inset: 0; }",
        "tokens": 140, "relevant": False,
    },
    "utils/email.py": {
        "content": "from email.mime.text import MIMEText\nimport smtplib\ndef send_welcome_email(user):\n    msg = MIMEText(f'Hello {user.name}, welcome!')\n    msg['Subject'] = 'Welcome!'\n    with smtplib.SMTP('smtp.gmail.com', 587) as s:\n        s.starttls()\n        s.send_message(msg)",
        "tokens": 110, "relevant": False,
    },
    "CHANGELOG.md": {
        "content": "# Changelog\n## v2.3.1 - Fixed pagination\n## v2.3.0 - Added dark mode\n## v2.2.0 - API rewrite",
        "tokens": 95, "relevant": False,
    },
    "tests/conftest.py": {
        "content": "import pytest\nfrom app import create_app\n@pytest.fixture\ndef app():\n    app = create_app('testing')\n    yield app\n@pytest.fixture\ndef client(app):\n    return app.test_client()",
        "tokens": 100, "relevant": False,
    },
    "utils/validators.py": {
        "content": "import re\ndef validate_email(email):\n    return bool(re.match(r'^[a-zA-Z0-9_.+-]+@', email))\ndef validate_phone(phone):\n    return bool(re.match(r'^\\+?[1-9]', phone))",
        "tokens": 90, "relevant": False,
    },
    "views/home.py": {
        "content": "from flask import render_template, jsonify\ndef render_homepage():\n    return render_template('index.html', title='Welcome')\ndef api_health():\n    return jsonify({'status': 'healthy'})",
        "tokens": 65, "relevant": False,
    },
    "auth/db_backup.py": {
        "content": "import sqlite3\ndef get_user(user_id):\n    conn = sqlite3.connect('app.db')\n    cursor = conn.cursor()\n    cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')\n    return cursor.fetchone()\ndef delete_user(user_id):\n    conn = sqlite3.connect('app.db')\n    cursor = conn.cursor()\n    cursor.execute(f'DELETE FROM users WHERE id = {user_id}')\n    conn.commit()",
        "tokens": 85, "relevant": False,
    },
    "deploy/docker-compose.yml": {
        "content": "version: '3.8'\nservices:\n  web:\n    build: .\n    ports: ['8000:8000']\n  db:\n    image: postgres:16\n  redis:\n    image: redis:7-alpine",
        "tokens": 90, "relevant": False,
    },
}

QUERY = "fix the SQL injection vulnerability in cursor.execute"
TOKEN_BUDGET = 250

engine = EntrolyEngine(
    w_recency=0.30, w_frequency=0.25, w_semantic=0.25, w_entropy=0.20,
    decay_half_life=15, min_relevance=0.01,
)

# Ingest all fragments
ingest_results = {}
for src, f in CODEBASE.items():
    t0 = time.perf_counter()
    result = dict(engine.ingest(f["content"], src, f["tokens"], False))
    elapsed_us = (time.perf_counter() - t0) * 1_000_000
    ingest_results[src] = {
        "status": result.get("status"),
        "entropy": result.get("entropy_score", 0),
        "fragment_id": result.get("fragment_id", result.get("duplicate_of", "")),
        "ingest_us": elapsed_us,
    }

# Optimize
t0 = time.perf_counter()
opt = dict(engine.optimize(TOKEN_BUDGET, QUERY))
opt_ms = (time.perf_counter() - t0) * 1000

selected = [dict(s) for s in opt.get("selected", [])]
stats = dict(engine.stats())

# Output as JSON
output = {
    "ingest_results": ingest_results,
    "optimize_ms": opt_ms,
    "selected": selected,
    "total_tokens_used": opt.get("total_tokens", 0),
    "skeleton_count": opt.get("skeleton_count", 0),
    "skeleton_tokens": opt.get("skeleton_tokens", 0),
    "stats": {
        "session": dict(stats.get("session", {})),
        "dedup": dict(stats.get("dedup", {})),
        "context_efficiency": dict(stats.get("context_efficiency", {})),
    },
    "codebase_info": {k: {"tokens": v["tokens"], "relevant": v["relevant"]} for k, v in CODEBASE.items()},
}
print(json.dumps(output, indent=2, default=str))
