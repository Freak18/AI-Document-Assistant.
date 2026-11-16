#!/usr/bin/env bash
# AI Tool - Single Click Launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
ENV_FILE="$BACKEND_DIR/.env"
PYTHON_BIN="${PYTHON_BIN:-$BACKEND_DIR/.venv/bin/python}"

FAST_MODE=0
FORCE_INSTALL=0

for arg in "$@"; do
  case "$arg" in
    --fast|--skip-tests|-f)
      FAST_MODE=1
      ;;
    --install|-i)
      FORCE_INSTALL=1
      ;;
    --help|-h)
      echo "Usage: ./start.sh [options]"
      echo ""
      echo "Options:"
      echo "  --fast, -f        Skip architectural verification test suite for instant startup"
      echo "  --install, -i     Force reinstallation of Python backend requirements"
      echo "  --help, -h        Show this help message"
      exit 0
      ;;
  esac
done

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  echo ""
  echo "Shutting down AI Tool services..."
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  # Clean up any leftover listeners
  for port in 8000 5173; do
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      kill -9 $pids 2>/dev/null || true
    fi
  done
  echo "Services stopped cleanly."
  exit "$exit_code"
}
trap cleanup INT TERM EXIT

echo "======================================================================"
echo "    AI Tool - Production Architecture Launcher"
echo "    FastAPI · LangGraph · pgvector · Deep Agent · React 19"
echo "======================================================================"

# 1. Check prerequisites
if ! command -v npm >/dev/null 2>&1; then
  echo "[ERROR] 'npm' not found. Please install Node.js 18+ before running the app."
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    echo "Creating backend/.env from .env.example..."
    cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
  elif [ -f "$BACKEND_DIR/.env.example" ]; then
    echo "Creating backend/.env from backend/.env.example..."
    cp "$BACKEND_DIR/.env.example" "$ENV_FILE"
  else
    echo "[ERROR] Missing backend/.env file. Please create it with DATABASE_URL and LLM keys."
    exit 1
  fi
fi

# 2. Python Virtual Environment
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Creating Python virtual environment at $BACKEND_DIR/.venv..."
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv "$BACKEND_DIR/.venv"
  elif command -v uv >/dev/null 2>&1; then
    uv venv "$BACKEND_DIR/.venv"
  else
    python3 -m venv "$BACKEND_DIR/.venv"
  fi
fi

# 3. Backend dependencies check
if [ "$FORCE_INSTALL" -eq 1 ] || ! "$PYTHON_BIN" -c "import fastapi, uvicorn, sqlalchemy, pgvector, langgraph" >/dev/null 2>&1; then
  echo "Installing / verifying backend dependencies from requirements.txt..."
  "$PYTHON_BIN" -m pip install -r "$BACKEND_DIR/requirements.txt"
else
  echo "✓ Backend dependencies verified."
fi

# 4. Load environment variables
set -a
source "$ENV_FILE"
set +a

if [ -z "${DATABASE_URL:-}" ]; then
  echo "[ERROR] DATABASE_URL is not set in backend/.env"
  exit 1
fi

# 5. Database detection & startup
DB_HOST="$($PYTHON_BIN -c 'import os; from urllib.parse import urlparse; print((urlparse(os.environ.get("DATABASE_URL", "")).hostname or "").lower())' 2>/dev/null || true)"

if [[ "$DB_HOST" == "localhost" || "$DB_HOST" == "127.0.0.1" || "$DB_HOST" == "postgres" || "$DB_HOST" == "db" ]]; then
  if command -v docker >/dev/null 2>&1; then
    echo "Detected local Postgres host ($DB_HOST); starting Docker Postgres container..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d postgres
    echo "Waiting for local Postgres container to be healthy..."
    until docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps postgres | grep -q "healthy\|running"; do
      sleep 1
    done
    echo "✓ Local PostgreSQL is ready."
  else
    echo "[ERROR] DATABASE_URL points to a local Postgres ($DB_HOST), but Docker is not available."
    exit 1
  fi
else
  echo "✓ Using remote PostgreSQL instance (${DB_HOST:-external})."
fi

# 6. Database schema bootstrap
echo "Bootstrapping database schema & vector extensions..."
cd "$BACKEND_DIR"
"$PYTHON_BIN" -c "from database import ensure_database_ready; ensure_database_ready(); print('✓ Database schema is up to date.')"
"$PYTHON_BIN" -m alembic upgrade head >/dev/null 2>&1 || true

# 7. Architecture tests
if [ "$FAST_MODE" -eq 0 ]; then
  echo "Running architectural verification tests (pass --fast to skip)..."
  "$PYTHON_BIN" test_architecture.py
  echo "✓ All architectural tests passed!"
else
  echo "ℹ Fast mode enabled: skipping architectural tests."
fi

# 8. Free any stale processes on ports 8000 and 5173
for port in 8000 5173; do
  pids=$(lsof -ti:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill -9 $pids 2>/dev/null || true
  fi
done
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# 9. Start Backend
echo "Starting FastAPI backend on http://127.0.0.1:8000..."
cd "$BACKEND_DIR"
DATABASE_URL="$DATABASE_URL" "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000 > "$SCRIPT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend health check..."
READY=0
for i in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [ "$READY" -ne 1 ]; then
  echo "[ERROR] Backend failed to start. Last log lines from backend.log:"
  tail -n 30 "$SCRIPT_DIR/backend.log" || true
  exit 1
fi
echo "✓ Backend is healthy."

# 10. Start Frontend
echo "Starting Vite React frontend on http://localhost:5173..."
cd "$FRONTEND_DIR"
if [ ! -d "node_modules" ]; then
  echo "Installing frontend dependencies..."
  npm install
fi

npm run dev -- --host 0.0.0.0 --port 5173 --strictPort > "$SCRIPT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo "Waiting for frontend to become ready..."
FRONTEND_READY=0
for i in $(seq 1 30); do
  if curl -fsS http://localhost:5173 >/dev/null 2>&1; then
    FRONTEND_READY=1
    break
  fi
  sleep 1
done

echo ""
echo "======================================================================"
echo "    🎉 AI Tool is LIVE!"
echo "======================================================================"
echo "  • Frontend UI:    http://localhost:5173"
echo "  • Backend API:    http://127.0.0.1:8000"
echo "  • API Docs:       http://127.0.0.1:8000/docs"
echo "  • Database:       ${DB_HOST:-PostgreSQL}"
echo "  • Logs:           backend.log / frontend.log"
echo "======================================================================"
echo "Press [Ctrl+C] to gracefully stop all services."
echo ""

# 11. Open browser
if command -v open >/dev/null 2>&1; then
  open "http://localhost:5173"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5173"
fi

wait
