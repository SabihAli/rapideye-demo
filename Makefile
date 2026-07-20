# RapidEye Demo — Makefile
# Cross-platform helpers (Windows + Linux/macOS). Uses venv Python directly — no shell activation needed.
#
# Python 3.11+ is required (see requirements.txt). Override interpreter:
#   make install PYTHON="py -3.13"
#   make install PYTHON=python3.12

PYTHON ?= python

ifeq ($(OS),Windows_NT)
	    VENV_PY  := .venv\Scripts\python.exe
	    RM_RF    := rmdir /s /q
	    NODE_BIN := node
	    NPM_CMD  := npm
else
	    VENV_PY  := .venv/bin/python
	    RM_RF    := rm -rf
	    NVM_NODE_DIR := $(shell bash -lc 'ls -d "$$HOME"/.nvm/versions/node/* 2>/dev/null | sort -V | tail -n1')
	    ifneq ($(strip $(NVM_NODE_DIR)),)
	        NODE_BIN := $(NVM_NODE_DIR)/bin/node
	        NPM_CMD  := $(NODE_BIN) $(NVM_NODE_DIR)/lib/node_modules/npm/bin/npm-cli.js
	        NODE_ENV_PREFIX := PATH="$(NVM_NODE_DIR)/bin:$$PATH"
	    else
	        NODE_BIN := node
	        NPM_CMD  := npm
	        NODE_ENV_PREFIX :=
	    endif
endif

.PHONY: help env check-python venv install install-web freeze backend frontend dev test clean

help:
	@echo "RapidEye Demo targets:"
	@echo "  make env          Copy .env.example to .env if missing"
	@echo "  make venv         Create Python 3.11+ virtual environment (.venv)"
	@echo "  make install      Install the exact pinned stack from requirements-lock.txt + editable package"
	@echo "  make install-web  Install frontend npm dependencies (npm ci, exact lock)"
	@echo "  make freeze       Regenerate requirements-lock.txt from the current .venv"
	@echo "  make backend      Run FastAPI backend (API_PORT from .env, default 8001)"
	@echo "  make frontend     Run Vite dev server on :5173"
	@echo "  make dev          Print instructions to run backend + frontend"
	@echo "  make test         Run backend pytest suite"
	@echo "  make clean        Remove .venv and web/node_modules"
	@printf '\n'
	@echo 'Python 3.11+ required. Override: make install PYTHON="py -3.13"'
	@echo "Frontend uses NODE=$(NODE_BIN)"

env:
ifeq ($(OS),Windows_NT)
	@if not exist .env copy .env.example .env
else
	@test -f .env || cp .env.example .env
endif
	@echo .env ready

check-python:
	@$(PYTHON) -c "import sys; v=sys.version_info; assert v>=(3,11), f'Python 3.11+ required, got {sys.version}. On Windows: make clean && make install PYTHON=py -3.13'"

venv: check-python
	$(PYTHON) -m venv .venv
	@echo Virtual environment created at .venv using $(PYTHON)

# InsightFace pulls plain `onnxruntime` which overwrites onnxruntime-gpu.
# After the lock install we uninstall the CPU package and force-reinstall GPU.
install: check-python env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) $(PYTHON) -m venv .venv
else
	@test -f $(VENV_PY) || $(PYTHON) -m venv .venv
endif
	@$(VENV_PY) -c "import sys; v=sys.version_info; assert v>=(3,11), f'.venv uses Python {sys.version} — run make clean then make install PYTHON=py -3.13'"
	$(VENV_PY) -m pip install -U "pip" "setuptools==81.0.0" "wheel"
	$(VENV_PY) -m pip install -r requirements-lock.txt
	-$(VENV_PY) -m pip uninstall -y onnxruntime
	$(VENV_PY) -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.24.4 tensorrt-cu12==10.13.3.9
	$(VENV_PY) -m pip install -e ".[dev]" --no-deps
	@echo "Backend dependencies installed (requirements-lock.txt + editable vision-demo)"

install-web:
ifeq ($(OS),Windows_NT)
	cd web && $(NPM_CMD) ci
else
	cd web && $(NODE_ENV_PREFIX) $(NPM_CMD) ci
endif
	@echo Frontend dependencies installed (npm ci — exact package-lock.json versions)

freeze: env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) (echo Run make install first && exit /b 1)
else
	@test -f $(VENV_PY) || (echo "Run make install first" && exit 1)
endif
	$(VENV_PY) scripts/generate_lock.py

backend: env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) (echo Run make install first && exit /b 1)
else
	@test -f $(VENV_PY) || (echo "Run make install first" && exit 1)
endif
	$(VENV_PY) -c "from server.config import settings; import os, sys; os.execvp(sys.executable, [sys.executable, '-m', 'uvicorn', 'server.main:app', '--host', settings.api_host, '--port', str(settings.api_port), '--reload', '--timeout-graceful-shutdown', '10'])"

frontend:
	@$(NODE_BIN) -e "const v=process.versions.node.split('.').map(Number); const ok=(v[0]>22)|| (v[0]===22&&v[1]>=12) || (v[0]===20&&v[1]>=19); if(!ok){console.error('Frontend requires Node 20.19+ or 22.12+. Current: '+process.versions.node); process.exit(1)}"
ifeq ($(OS),Windows_NT)
	cd web && set CHOKIDAR_USEPOLLING=1&& set CHOKIDAR_INTERVAL=300&& $(NPM_CMD) run dev
else
	cd web && CHOKIDAR_USEPOLLING=1 CHOKIDAR_INTERVAL=300 $(NODE_ENV_PREFIX) $(NPM_CMD) run dev
endif

dev:
	@echo Run in two terminals:
	@echo   Terminal 1: make backend
	@echo   Terminal 2: make frontend
	@echo Then open http://localhost:5173

test: env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) (echo Run make install first && exit /b 1)
else
	@test -f $(VENV_PY) || (echo "Run make install first" && exit 1)
endif
	$(VENV_PY) -m pytest tests/ -v

clean:
ifeq ($(OS),Windows_NT)
	@if exist .venv $(RM_RF) .venv
	@if exist web\node_modules $(RM_RF) web\node_modules
else
	$(RM_RF) .venv web/node_modules
endif
