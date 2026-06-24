#!/bin/bash
set -e
sudo apt-get update -y
sudo apt-get install -y ffmpeg wget unzip git python3-venv python3-pip
echo "voiceguard stage2 base deps installed"
