#!/usr/bin/env bash
# setup_syncthing_lumi.sh — Install Syncthing on LUMI and print pairing instructions.
#
# Syncthing syncs outputs/track_b/ between your local machine and LUMI automatically.
# New experiment folders (e.g. 19130485_full/) appear locally as soon as a LUMI job writes them.
# Uses an SSH tunnel — no firewall ports need to be opened on LUMI.
#
# Usage (run from local machine):
#   bash scripts/setup_syncthing_lumi.sh

set -euo pipefail

LUMI_USER="sousapoz"
LUMI_HOST="lumi.csc.fi"
SSH_KEY="$HOME/.ssh/id_rsa_lumi"
SCRATCH="/scratch/project_465002610/$LUMI_USER"
SYNCTHING_VERSION="1.28.0"   # update as needed

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i $SSH_KEY $LUMI_USER@$LUMI_HOST"

echo "========================================"
echo "Syncthing setup for LUMI"
echo "========================================"

# ── Step 0: Sync updated code to LUMI ─────────────────────────────────────
echo
echo ">>> Step 0/4: Rsyncing code to LUMI..."
rsync -av --exclude='outputs/' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.git/' --exclude='.venv/' \
    -e "ssh -i $SSH_KEY" \
    "$REPO/" "$LUMI_USER@$LUMI_HOST:$SCRATCH/code/"
echo "Code synced."

# ── Step 1: Install Syncthing on LUMI ─────────────────────────────────────
echo
echo ">>> Installing Syncthing $SYNCTHING_VERSION on LUMI..."
$SSH "
  if [ -f $SCRATCH/bin/bin/syncthing ]; then
    echo 'Syncthing already installed:'
    $SCRATCH/bin/bin/syncthing --version
  else
    cd $SCRATCH
    curl -fsSL https://github.com/syncthing/syncthing/releases/download/v${SYNCTHING_VERSION}/syncthing-linux-amd64-v${SYNCTHING_VERSION}.tar.gz \
        | tar -xz
    cp syncthing-linux-amd64-v${SYNCTHING_VERSION}/syncthing $SCRATCH/bin/bin/syncthing
    rm -rf syncthing-linux-amd64-v${SYNCTHING_VERSION}
    echo 'Installed:'
    $SCRATCH/bin/bin/syncthing --version
  fi
"

# ── Step 2: Generate LUMI device ID ───────────────────────────────────────
echo
echo ">>> Step 2/4: Getting LUMI Syncthing device ID..."
LUMI_DEVICE_ID=$($SSH "$SCRATCH/bin/bin/syncthing --device-id 2>/dev/null || true")
echo "LUMI device ID: $LUMI_DEVICE_ID"

# ── Step 3: Create LUMI Syncthing config ──────────────────────────────────
echo
echo ">>> Step 3/4: Writing Syncthing config on LUMI..."
$SSH "
  mkdir -p $SCRATCH/syncthing_config
  cat > $SCRATCH/syncthing_config/config.xml <<'XMLEOF'
<configuration version=\"37\">
    <folder id=\"bioreasonb-outputs\" label=\"BioReasoning Track B outputs\" path=\"$SCRATCH/outputs/track_b\" type=\"sendreceive\">
        <filesystemType>basic</filesystemType>
        <rescanIntervalS>30</rescanIntervalS>
        <fsWatcherEnabled>true</fsWatcherEnabled>
        <ignorePerms>false</ignorePerms>
        <autoNormalize>true</autoNormalize>
    </folder>
    <gui enabled=\"true\" tls=\"false\" debugging=\"false\">
        <address>127.0.0.1:8384</address>
    </gui>
    <options>
        <listenAddress>tcp://127.0.0.1:22000</listenAddress>
        <globalAnnounceEnabled>false</globalAnnounceEnabled>
        <localAnnounceEnabled>false</localAnnounceEnabled>
        <relaysEnabled>false</relaysEnabled>
        <natEnabled>false</natEnabled>
        <urAccepted>-1</urAccepted>
    </options>
</configuration>
XMLEOF
  echo 'Config written.'
"

# ── Step 4: Create LUMI start/stop wrapper ────────────────────────────────
echo
echo ">>> Step 4/4: Writing syncthing_start.sh on LUMI..."
$SSH "cat > $SCRATCH/syncthing_start.sh << 'EOF'
#!/bin/bash
# Run Syncthing on the LUMI login node.
# Keep this running in a tmux/screen session while you want sync active.
export HOME_CONFIG=$SCRATCH/syncthing_config
$SCRATCH/bin/bin/syncthing --home=\$HOME_CONFIG --no-browser
EOF
chmod +x $SCRATCH/syncthing_start.sh
echo 'Done.'"

# ── Step 5: Print local setup instructions ────────────────────────────────
echo
echo "========================================"
echo "Setup complete. Follow these steps:"
echo "========================================"
echo
echo "1. Start Syncthing on LUMI (in a tmux session so it persists):"
echo
echo "   ssh -i $SSH_KEY $LUMI_USER@$LUMI_HOST"
echo "   tmux new -s sync"
echo "   bash $SCRATCH/syncthing_start.sh"
echo "   # Ctrl-B D  to detach"
echo
echo "2. Open an SSH tunnel from your LOCAL machine (keep this terminal open):"
echo
echo "   ssh -i $SSH_KEY -N \\"
echo "       -L 127.0.0.1:22001:127.0.0.1:22000 \\"
echo "       $LUMI_USER@$LUMI_HOST"
echo
echo "   Or use autossh to auto-reconnect:"
echo "   autossh -M 0 -N -i $SSH_KEY \\"
echo "       -L 127.0.0.1:22001:127.0.0.1:22000 \\"
echo "       $LUMI_USER@$LUMI_HOST &"
echo
echo "3. In your LOCAL Syncthing (http://localhost:8384):"
echo "   a. Add device:  tcp://127.0.0.1:22001"
echo "      Device ID:   $LUMI_DEVICE_ID"
echo "   b. Share folder 'BioReasoning Track B outputs'"
echo "      Local path:  $(cd "$(dirname "$0")/.." && pwd)/outputs/track_b"
echo "      Folder ID:   bioreasonb-outputs"
echo
echo "4. That's it. New experiment folders (e.g. 19130485_full/) will appear"
echo "   in your local outputs/track_b/ automatically after each LUMI job."
echo
echo "To stop sync on LUMI:"
echo "   ssh lumi 'tmux kill-session -t sync'"
echo "========================================"
