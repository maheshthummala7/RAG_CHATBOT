#!/usr/bin/env bash
set -e

echo "=========================================="
echo " Starting Deployment: $(date)"
echo "=========================================="

echo "=== Pulling latest changes from GitHub ==="
git fetch origin master
git reset --hard origin/master

echo "=== Checking virtual environment ==="
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv..."
    if command -v python3.10 &>/dev/null; then
        python3.10 -m venv .venv
    else
        python3 -m venv .venv
    fi
fi

echo "=== Activating virtual environment ==="
source .venv/bin/activate

echo "=== Installing / Updating dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Checking environment configuration ==="
if [ ! -f ".env/.env" ] && [ ! -f ".env" ]; then
    echo "WARNING: No .env/.env or .env file found. Make sure your API keys are configured."
fi

echo "=== Restarting rag-chatbot service ==="
if systemctl list-unit-files | grep -q "rag-chatbot.service"; then
    sudo systemctl restart rag-chatbot
    sleep 3
    if sudo systemctl is-active --quiet rag-chatbot; then
        echo "=== rag-chatbot service successfully restarted and active! ==="
        sudo systemctl status rag-chatbot --no-pager -n 10
    else
        echo "ERROR: rag-chatbot service failed to start. Showing last 30 log lines:"
        sudo journalctl -u rag-chatbot -n 30 --no-pager
        exit 1
    fi
else
    echo "Notice: rag-chatbot.service is not installed in /etc/systemd/system/ yet."
    echo "Please copy rag-chatbot.service.example to /etc/systemd/system/rag-chatbot.service and enable it."
fi

echo "=========================================="
echo " Deployment Completed Successfully: $(date)"
echo "=========================================="
