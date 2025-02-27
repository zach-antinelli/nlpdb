#!/usr/bin/env bash

set -euo pipefail

log_message() {
    LVL="${1:-}" MSG="${2:-}"
    TS=$(date +'%Y-%m-%d %H:%M:%S')
    printf "%s\t%s\t%s\n" "$LVL" "$TS" "$MSG"
    if [[ "$LVL" = "ERROR" ]]; then
        exit 1
    fi
}

kill_app() {
    if pkill -f "python app.py"; then
        log_message INFO "Stopped existing application process"
    else
        log_message INFO "No existing application process found"
    fi
}

kill_app

if [[ "${1:-}" == "-k" ]]; then
    exit 0
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    log_message ERROR "Virtual environment not active"
    exit 1
fi

if ! command -v flask >/dev/null 2>&1; then
    log_message ERROR "Flask is not installed"
    exit 1
fi

log_message INFO "Starting application..."
if nohup python app.py >>app.log 2>&1 & then
    PID=$!
    sleep 2
    if ps -p $PID >/dev/null; then
        log_message INFO "Application started successfully (PID: $PID)"
        log_message INFO "Logs available in app.log"
        log_message INFO "Application available at http://127.0.0.1:8080"
    else
        log_message ERROR "Application failed to start"
        exit 1
    fi
else
    log_message ERROR "Failed to start application, see app.log"
    exit 1
fi
