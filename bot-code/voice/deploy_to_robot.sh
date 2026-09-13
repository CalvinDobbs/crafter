#!/usr/bin/env bash
# Deploy voice-enabled mc_skills to the robot.
# Usage:
#   1. Remote via SSH: ./bot-code/voice/deploy_to_robot.sh bracketbot@100.66.148.86
#   2. Local on robot: ./deploy_to_robot.sh /home/bracketbot/bbapps/mc_skills

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-/home/bracketbot/bbapps/mc_skills}"
VOICE_PY=(hook.py moves.py narrator.py speech_queue.py tts.py packs.py flavor.py from_plan.py)

echo "=== Deploying Voice to mc_skills ==="
echo "Source: $SCRIPT_DIR"

if [[ "$TARGET" =~ @ ]]; then
    HOST="$TARGET"
    REMOTE_DIR="/home/bracketbot/bbapps/mc_skills"
    echo "Deploying remotely to $HOST:$REMOTE_DIR..."

    ssh -o BatchMode=yes "$HOST" "mkdir -p $REMOTE_DIR/wavs && [ -f $REMOTE_DIR/main.py ] && cp $REMOTE_DIR/main.py $REMOTE_DIR/main.py.bak || true"

    scp "$SCRIPT_DIR/mc_skills_main.py" "$HOST:$REMOTE_DIR/main.py"
    for f in "${VOICE_PY[@]}"; do
        scp "$SCRIPT_DIR/$f" "$HOST:$REMOTE_DIR/"
    done
    scp "$SCRIPT_DIR/wavs/"*.wav "$HOST:$REMOTE_DIR/wavs/"

    echo "[+] Remote deployment complete."
    echo "Start: ssh $HOST 'cd $REMOTE_DIR && VOICE_PACK=boxing ~/.local/bin/uv run main.py'"
else
    DEST_DIR="$TARGET"
    echo "Deploying locally to $DEST_DIR..."
    mkdir -p "$DEST_DIR/wavs"
    if [ -f "$DEST_DIR/main.py" ]; then
        cp "$DEST_DIR/main.py" "$DEST_DIR/main.py.bak"
    fi
    cp "$SCRIPT_DIR/mc_skills_main.py" "$DEST_DIR/main.py"
    for f in "${VOICE_PY[@]}"; do
        cp "$SCRIPT_DIR/$f" "$DEST_DIR/"
    done
    cp "$SCRIPT_DIR/wavs/"*.wav "$DEST_DIR/wavs/"
    echo "[+] Local deployment to $DEST_DIR complete."
fi
