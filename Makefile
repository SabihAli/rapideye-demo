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
else
    VENV_PY  := .venv/bin/python
    RM_RF    := rm -rf
endif

.PHONY: help env check-python venv install install-web backend frontend dev test clean

help:
	@echo RapidEye Demo targets:
	@echo   make env          Copy .env.example to .env if missing
	@echo   make venv         Create Python 3.11+ virtual environment (.venv)
	@echo   make install      Install pinned deps from requirements.txt + editable package
	@echo   make install-web  Install frontend npm dependencies
	@echo   make backend      Run FastAPI backend (API_PORT from .env, default 8001)
	@echo   make frontend     Run Vite dev server on :5173
	@echo   make dev          Print instructions to run backend + frontend
	@echo   make test         Run backend pytest suite
	@echo   make clean        Remove .venv and web/node_modules
	@echo.
	@echo Python 3.11+ required. Override: make install PYTHON="py -3.13"

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

install: check-python env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) $(PYTHON) -m venv .venv
else
	@test -f $(VENV_PY) || $(PYTHON) -m venv .venv
endif
	@$(VENV_PY) -c "import sys; v=sys.version_info; assert v>=(3,11), f'.venv uses Python {sys.version} — run make clean then make install PYTHON=py -3.13'"
	$(VENV_PY) -m pip install -U pip setuptools wheel
	$(VENV_PY) -m pip install -r requirements.txt
	$(VENV_PY) -m pip install -e ".[dev]" --no-deps
	@echo Backend dependencies installed (requirements.txt + editable vision-demo)

install-web:
	cd web && npm install
	@echo Frontend dependencies installed

backend: env
ifeq ($(OS),Windows_NT)
	@if not exist $(VENV_PY) (echo Run make install first && exit /b 1)
else
	@test -f $(VENV_PY) || (echo "Run make install first" && exit 1)
endif
	$(VENV_PY) -c "from server.config import settings; import subprocess, sys; raise SystemExit(subprocess.call([sys.executable, '-m', 'uvicorn', 'server.main:app', '--host', settings.api_host, '--port', str(settings.api_port), '--reload']))"

frontend:
	cd web && npm run dev

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
