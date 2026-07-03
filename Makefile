# RapidEye Demo — Makefile
# Cross-platform helpers (Windows + Linux/macOS). Uses venv Python directly — no shell activation needed.

ifeq ($(OS),Windows_NT)
    VENV_PY  := .venv\Scripts\python.exe
    VENV_PIP := .venv\Scripts\pip.exe
    RM_RF    := rmdir /s /q
else
    VENV_PY  := .venv/bin/python
    VENV_PIP := .venv/bin/pip
    RM_RF    := rm -rf
endif

.PHONY: help env venv install install-web backend frontend dev test clean

help:
	@echo RapidEye Demo targets:
	@echo   make env          Copy .env.example to .env if missing
	@echo   make venv         Create Python virtual environment (.venv)
	@echo   make install      Install backend Python dependencies
	@echo   make install-web  Install frontend npm dependencies
	@echo   make backend      Run FastAPI backend on :8000
	@echo   make frontend     Run Vite dev server on :5173
	@echo   make dev          Print instructions to run backend + frontend
	@echo   make test         Run backend pytest suite
	@echo   make clean        Remove .venv and web/node_modules

env:
ifeq ($(OS),Windows_NT)
	@if not exist .env copy .env.example .env
else
	@test -f .env || cp .env.example .env
endif
	@echo .env ready

venv:
	python -m venv .venv
	@echo Virtual environment created at .venv

install: venv env
	$(VENV_PIP) install -U pip
	$(VENV_PIP) install -e ".[dev]"
	@echo Backend dependencies installed

install-web:
	cd web && npm install
	@echo Frontend dependencies installed

backend: env
	$(VENV_PY) -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	cd web && npm run dev

dev:
	@echo Run in two terminals:
	@echo   Terminal 1: make backend
	@echo   Terminal 2: make frontend
	@echo Then open http://localhost:5173

test: env
	$(VENV_PY) -m pytest tests/ -v

clean:
ifeq ($(OS),Windows_NT)
	@if exist .venv $(RM_RF) .venv
	@if exist web\node_modules $(RM_RF) web\node_modules
else
	$(RM_RF) .venv web/node_modules
endif
