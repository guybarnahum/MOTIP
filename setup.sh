#!/usr/bin/env bash
#
# Setup script for MOTIP (AWS/GPU)
# - Finds Python 3.12 (preferred) or 3.10+
# - Installs System Deps (Ninja, build-essential, ffmpeg)
# - Installs PyTorch 2.4.0 (cu121)
# - Installs MOTIP dependencies + opencv
# - Compiles Deformable Attention (CUDA ops)
# - Verifies installation with a lightweight import check
#
set -e

# ---------------- Auto-yes handling ----------------
AUTO_YES=""
for arg in "$@"; do
  case "$arg" in
    --yes|-y) AUTO_YES="--yes" ;;
    *) ;;
  esac
done

ask_yes_no() {
  local prompt="$1"
  if [[ -n "$AUTO_YES" ]]; then return 0; fi
  read -p "$prompt [y/N] " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]]
}

# ---------------- Colors & Spinner ----------------
if tput setaf 0 >/dev/null 2>&1; then
  COLOR_GRAY="$(tput setaf 8)"
  COLOR_RESET="$(tput sgr0)"
else
  COLOR_GRAY=$'\033[90m'
  COLOR_RESET=$'\033[0m'
fi

run_and_log() {
  local log_file; log_file=$(mktemp)
  local description="$1"; shift

  # Hide cursor
  tput civis 2>/dev/null || true

  local prev_render=""
  local cols; cols=$(tput cols 2>/dev/null || echo 120)

  (
    frames=( '⠋ ' '⠙ ' '⠹ ' '⠸ ' '⠼ ' '⠴ ' '⠦ ' '⠧ ' '⠇ ' '⠏ ' )
    i=0
    while :; do
      local last_line=""
      if [[ -s "$log_file" ]]; then
        # tail the last line, strip ANSI codes, and remove carriage returns that confuse the terminal
        last_line=$(tail -n 1 "$log_file" | sed -E 's/\x1B\[[0-9;?]*[ -/]*[@-~]//g' | tr -d '\r')
      fi

      local prefix="${frames[i]}${description} : "
      
      # Calculate available space for the log tail
      local available_space=$((cols - ${#prefix} - 1))
      if (( available_space < 0 )); then available_space=0; fi

      # Truncate last_line if it's too long
      if (( ${#last_line} > available_space )); then
        last_line="${last_line:0:$available_space}"
      fi

      # Construct the visual string
      # We construct "head" (white) and "tail" (gray) separately
      local render="${COLOR_RESET}${prefix}${COLOR_GRAY}${last_line}${COLOR_RESET}"

      # Only print if changed to reduce flicker
      if [[ "$render" != "$prev_render" ]]; then
        # \033[K clears the line from cursor to end
        printf '\r\033[K%s' "$render"
        prev_render="$render"
      fi

      i=$(( (i+1) % ${#frames[@]} ))
      sleep 0.2
    done
  ) &
  local spinner_pid=$!

  # Run the command
  if ! "$@" >"$log_file" 2>&1; then
    kill "$spinner_pid" &>/dev/null || true
    wait "$spinner_pid" &>/dev/null || true
    # Restore cursor
    tput cnorm 2>/dev/null || true
    
    printf '\r\033[K%s' "${COLOR_RESET}"
    printf "❌ %s failed.\n" "$description"
    echo "--- ERROR LOG ---"
    cat "$log_file"
    echo "--- END LOG ---"
    rm -f "$log_file"
    exit 1
  fi

  kill "$spinner_pid" &>/dev/null || true
  wait "$spinner_pid" &>/dev/null || true
  # Restore cursor
  tput cnorm 2>/dev/null || true

  printf '\r\033[K%s' "${COLOR_RESET}"
  printf '✅ %s\n' "$description"
  rm -f "$log_file"
}

cleanup_render() { tput cnorm 2>/dev/null || true; }
trap cleanup_render EXIT INT TERM

# ---------------- Step 1: Environment Checks ----------------
echo "🔎 Checking Environment..."

# Find Python
if command -v python3.12 &>/dev/null; then PYTHON_BIN="python3.12"
elif command -v python3.10 &>/dev/null; then PYTHON_BIN="python3.10"
elif command -v python3 &>/dev/null; then PYTHON_BIN="python3"
else echo "❌ Python not found."; exit 1; fi
echo "   Using: $($PYTHON_BIN --version)"

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
  echo "   GPU Detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
else
  echo "⚠️  No GPU detected. Compilation of Deformable DETR ops will FAIL."
  if ! ask_yes_no "Continue anyway?"; then exit 1; fi
fi

# ---------------- Step 2: System Dependencies ----------------
if [[ "$(uname -s)" == "Linux" ]]; then
  # Added ffmpeg and libx264-dev for video processing
  if ! command -v ninja &>/dev/null || ! command -v ffmpeg &>/dev/null; then
     run_and_log "Install System Deps (ffmpeg, ninja, g++)" sudo apt-get update && sudo apt-get install -y ffmpeg libx264-dev ninja-build build-essential g++
  fi
  
  # Ensure CUDA_HOME is set for compilation
  if [ -d "/usr/local/cuda" ] && [ -z "$CUDA_HOME" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
    echo "   Exported CUDA_HOME=${CUDA_HOME}"
  fi
fi

# ---------------- Step 3: Virtual Environment ----------------
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  run_and_log "Create venv at ${VENV_DIR}" "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
run_and_log "Upgrade pip" pip install --upgrade pip wheel setuptools

# ---------------- Step 4: Install PyTorch ----------------
echo "📦 Installing PyTorch 2.4.0 (cu121)..."
# Using --no-cache-dir to prevent memory issues during pip install on smaller instances
run_and_log "Install PyTorch" pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# ---------------- Step 5: Install Python Dependencies ----------------
echo "📦 Installing MOTIP Dependencies..."
# Added opencv-python here
run_and_log "Install Deps" pip install pyyaml tqdm matplotlib scipy pandas wandb accelerate einops "numpy<2" toml opencv-python accelerate matplotlib

# ---------------- Step 6: Compile Custom Ops ----------------
echo "⚙️  Compiling Deformable Attention Ops..."

if [ -d "models/ops" ]; then
  cd models/ops
  # Build and Install
  run_and_log "Compile MSDeformAttn" python setup.py build install
  
  # Validate
  run_and_log "Test Compilation" python test.py
  cd ../..
else
  echo "❌ 'models/ops' directory not found. Are you in the MOTIP root?"
  exit 1
fi

# ---------------- Step 7: Finalize ----------------
if [ -f ~/.bashrc ] && ! grep -q "alias venv=" ~/.bashrc; then
    echo "alias venv='source .venv/bin/activate'" >> ~/.bashrc
fi

echo ""
echo "✅ Setup Complete!"
echo "   Activate:  source .venv/bin/activate"
echo "   Run:       python launcher.py"