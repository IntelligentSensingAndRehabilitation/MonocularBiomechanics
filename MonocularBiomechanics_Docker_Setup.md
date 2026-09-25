# MonocularBiomechanics Docker Setup Notes

These notes document the working setup for running **MonocularBiomechanics** on **Windows 11 Pro + WSL2 + Docker Desktop + NVIDIA RTX 5070**, including build, run, stop, restart, GPU verification, persistent results, and troubleshooting commands.

---

## 1. Target setup

```text
Windows 11 Pro
│
├── NVIDIA Windows driver
│
├── WSL2 (Ubuntu)
│
├── Docker Desktop
│   └── WSL2 backend
│
└── VS Code
    └── WSL extension

Ubuntu / WSL2
└── ~/projects/MonocularBiomechanics
    ├── Dockerfile
    ├── .dockerignore
    ├── main.py
    ├── pyproject.toml
    ├── uv.lock
    ├── monocular_demos/
    └── results/
```

The application runs in a Linux Docker container while VS Code runs on Windows and connects to the WSL filesystem.

---

## 2. One-time Windows / WSL setup

### Check WSL

Run in **Windows PowerShell**:

```powershell
wsl --version
wsl -l -v
```

Make sure Ubuntu uses WSL2.

If needed:

```powershell
wsl --set-version Ubuntu 2
```

Update WSL:

```powershell
wsl --update
```

To completely restart WSL later:

```powershell
wsl --shutdown
```

---

## 3. Check NVIDIA GPU in WSL

Open Ubuntu / WSL and run:

```bash
nvidia-smi
```

The RTX 5070 should appear.

Do **not** install a separate Linux NVIDIA display driver inside WSL.

---

## 4. Docker Desktop setup

Install Docker Desktop for Windows.

In Docker Desktop:

```text
Settings
→ General
→ Use the WSL 2 based engine
```

Then:

```text
Settings
→ Resources
→ WSL Integration
→ Enable Ubuntu
```

Apply and restart Docker Desktop.

---

## 5. Check Docker from WSL

```bash
docker --version
```

Test Docker:

```bash
docker run hello-world
```

If you get a Docker socket permission error, add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
```

Then close Ubuntu and run in Windows PowerShell:

```powershell
wsl --shutdown
```

Reopen Ubuntu and check:

```bash
groups
```

The `docker` group should be listed.

---

## 6. Clone the repository inside WSL

Keep the repo in the Linux filesystem rather than under `/mnt/c/...`.

```bash
mkdir -p ~/projects
cd ~/projects

git clone https://github.com/Xiong-biomech/MonocularBiomechanics.git
cd MonocularBiomechanics
```

Open it in VS Code:

```bash
code .
```

VS Code should show something similar to:

```text
WSL: Ubuntu
```

---

## 7. Useful VS Code extensions

Recommended:

- WSL
- Docker
- Python
- GitHub Copilot
- GitHub Copilot Chat
- Dev Containers (optional)

---

## 8. Recommended Git branch for Docker work

```bash
git checkout -b docker-support
```

Check:

```bash
git status
```

---

## 9. Create a persistent results directory

From the repo root:

```bash
mkdir -p results
```

This folder will be bind-mounted into Docker so `.npz` files survive container restarts.

Recommended `main.py` configuration:

```python
import os

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

Save keypoint and fitted-model outputs under `OUTPUT_DIR`.

Example:

```python
with open(os.path.join(OUTPUT_DIR, f"{fname}_keypoints.npz"), "wb") as f:
    ...
```

```python
with open(os.path.join(OUTPUT_DIR, f"{fname}_fitted_model.npz"), "wb") as f:
    ...
```

---

## 10. Dockerfile notes

The project uses:

- Ubuntu 24.04
- Python managed by `uv`
- JAX GPU support
- TensorFlow GPU support
- MuJoCo / MJX
- EGL / PyOpenGL
- FFmpeg

Important environment variables:

```dockerfile
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
```

Required Ubuntu packages include:

```dockerfile
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    ffmpeg \
    libgl1 \
    libegl1 \
    libgles2 \
    libosmesa6-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

Install dependencies:

```dockerfile
RUN uv sync --frozen --extra cuda
```

For TensorFlow GPU detection, the NVIDIA libraries installed inside the Python environment need to be included in `LD_LIBRARY_PATH`.

Example:

```dockerfile
ENV LD_LIBRARY_PATH="/app/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cufft/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/curand/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:\
/app/.venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib"
```

Expose `ptxas`:

```dockerfile
RUN ln -sf \
    /app/.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas \
    /app/.venv/bin/ptxas
```

Gradio should listen on all interfaces:

```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)
```

---

## 11. Build the Docker image

From the repo root:

```bash
docker build -t monocular-biomechanics .
```

For a completely clean rebuild:

```bash
docker build --no-cache -t monocular-biomechanics .
```

Use `--no-cache` only when necessary because dependency installation can take several minutes.

Check the image:

```bash
docker images
```

---

## 12. Verify GPU passthrough

### NVIDIA GPU inside the container

```bash
docker run --rm --gpus all \
  monocular-biomechanics \
  nvidia-smi
```

The RTX 5070 should appear.

### Test JAX

```bash
docker run --rm --gpus all \
  monocular-biomechanics \
  uv run python -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

Expected result:

```text
[CudaDevice(id=0)]
gpu
```

### Test TensorFlow

```bash
docker run --rm --gpus all \
  monocular-biomechanics \
  uv run python -c \
  "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected result:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

On the RTX 5070, TensorFlow may print a warning that compute capability `12.0a` is not included as a precompiled kernel target and PTX kernels may be JIT compiled. This can make the first run much slower.

---

## 13. Run the application with persistent results and GPU caches

For this project, it is useful to persist both generated results and GPU compilation caches. This is especially important on an RTX 5070 because TensorFlow may JIT-compile PTX kernels for compute capability 12.0a on first use, and JAX also uses a compilation cache.

Create the host results directory once:

```bash
mkdir -p results
```

Recommended run command:

```bash
docker run \
    --name monocular-test \
    --gpus all \
    -p 7860:7860 \
    -e OUTPUT_DIR=/app/results \
    -v "$(pwd)/results:/app/results" \
    -v monocular-jax-cache:/app/.jax_cache \
    -v monocular-cuda-cache:/root/.cache/cuda \
    monocular-biomechanics
```

This preserves:

```text
Host repo/results/          -> generated .npz files and other outputs
monocular-jax-cache         -> JAX compilation cache
monocular-cuda-cache        -> CUDA/PTX compilation cache
```

The named Docker volumes survive container removal and recreation.

Check them with:

```bash
docker volume ls
```

You should eventually see entries such as:

```text
monocular-jax-cache
monocular-cuda-cache
```

Add the CUDA cache settings to the Dockerfile so CUDA uses the persistent location:

```dockerfile
ENV CUDA_CACHE_PATH=/root/.cache/cuda
ENV CUDA_CACHE_MAXSIZE=2147483648
```

Then open the app in Windows:

```text
http://localhost:7860
```

Then open in Windows:

```text
http://localhost:7860
```

The host folder:

```text
~/projects/MonocularBiomechanics/results
```

is mapped to:

```text
/app/results
```

inside the container.

---

## 14. Check generated data

From another WSL terminal:

```bash
ls -lh ~/projects/MonocularBiomechanics/results
```

Typical output files:

```text
<video>_keypoints.npz
<video>_fitted_model.npz
```

Check continuously:

```bash
watch -n 1 'ls -lh ~/projects/MonocularBiomechanics/results'
```

---

## 15. Monitor the GPU while processing

Open another WSL terminal:

```bash
watch -n 1 nvidia-smi
```

This is useful during:

- Metrabs / TensorFlow keypoint detection
- JAX biomechanical fitting
- first-run PTX compilation

---

## 16. List Docker containers

Running containers only:

```bash
docker ps
```

All containers, including stopped ones:

```bash
docker ps -a
```

Filter by name:

```bash
docker ps -a --filter name=monocular-test
```

---

## 17. Stop the running container

From another terminal:

```bash
docker stop monocular-test
```

Or press:

```text
Ctrl+C
```

in the terminal where `docker run` is running.

Because the container was created **without `--rm`**, it remains available for inspection after stopping.

---

## 18. Start an existing stopped container again

If the container already exists and you have not rebuilt the image, you do **not** need to run `docker run` again. Start the same container:

```bash
docker start monocular-test
```

To start and attach to its console output:

```bash
docker start -a monocular-test
```

This preserves the container filesystem. The named JAX and CUDA cache volumes also remain available, so first-run compilation work does not need to be repeated from scratch.

Normal day-to-day sequence:

```bash
# Start existing container and attach
docker start -a monocular-test

# When finished
docker stop monocular-test
```

Only recreate the container when you change the Docker image or need different mounts/environment variables. If you rebuilt the image, remove the old container and create a new one instead of restarting it.

---

## 19. Remove a stopped container

```bash
docker rm monocular-test
```

Force-remove it even if running:

```bash
docker rm -f monocular-test
```

This is useful when Docker says:

```text
Conflict. The container name "/monocular-test" is already in use
```

Typical sequence:

```bash
docker rm -f monocular-test

docker run \
    --name monocular-test \
    --gpus all \
    -p 7860:7860 \
    -e OUTPUT_DIR=/app/results \
    -v "$(pwd)/results:/app/results" \
    -v monocular-jax-cache:/app/.jax_cache \
    -v monocular-cuda-cache:/root/.cache/cuda \
    monocular-biomechanics
```

---

## 20. Kill a Docker container immediately

Graceful stop:

```bash
docker stop monocular-test
```

Immediate kill:

```bash
docker kill monocular-test
```

Force removal:

```bash
docker rm -f monocular-test
```

Use `docker stop` normally. Use `docker kill` only when the container is unresponsive.

---

## 21. Inspect why a container stopped

Check status:

```bash
docker inspect monocular-test \
  --format='Status={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} Error={{.State.Error}}'
```

Examples:

```text
Status=exited
ExitCode=1
OOMKilled=false
```

means the program exited with an error but was not killed by memory pressure.

If:

```text
OOMKilled=true
```

the process was killed because it ran out of available memory.

---

## 22. View Docker logs

Full logs:

```bash
docker logs monocular-test
```

Last 100 lines:

```bash
docker logs --tail 100 monocular-test
```

Follow logs live:

```bash
docker logs -f monocular-test
```

Last 200 lines and continue following:

```bash
docker logs --tail 200 -f monocular-test
```

---

## 23. Open a shell inside a running container

```bash
docker exec -it monocular-test bash
```

Inside the container you can inspect:

```bash
pwd
ls -lah
ls -lah /app/results
nvidia-smi
```

Exit:

```bash
exit
```

---

## 24. Run a temporary debug container

Useful when testing the image without starting Gradio:

```bash
docker run --rm -it --gpus all \
  monocular-biomechanics \
  bash
```

Because of `--rm`, this temporary container is deleted automatically when you exit.

Use this for diagnostics only, not for important result files unless a host directory is mounted.

---

## 25. Validate Python syntax before rebuilding

Before rebuilding after editing `main.py`:

```bash
python3 -m py_compile main.py
```

No output means the file passed the syntax check.

If there is an indentation or syntax error, Python will show the line number.

To inspect numbered lines:

```bash
nl -ba main.py | head -50
```

---

## 26. Common Gradio output mismatch bug

If Gradio says:

```text
returned too many output values
```

check that the function returns the same number of values as the Gradio callback expects.

For example, if the callback is:

```python
outputs=[viz_info, viz_plot1]
```

then the function must return exactly two values:

```python
return result_text, plot1
```

and early returns should also return two values:

```python
return "Please select a fitted model file first.", None
```

not:

```python
return "Please select a fitted model file first.", None, None
```

---

## 27. Current output-path rule

Keypoints and fitted models should be loaded from `OUTPUT_DIR`.

Correct pattern:

```python
keypoints_file = os.path.join(
    OUTPUT_DIR,
    f"{fname}_keypoints.npz"
)

biomech_file = os.path.join(
    OUTPUT_DIR,
    selected_file
)
```

This keeps all generated data under `/app/results`, which is mounted to the host `results/` directory.

---

## 28. Important uploaded-video limitation

The fitted `.npz` currently stores the uploaded Gradio video path:

```python
video_path=video_path
```

Gradio upload paths are temporary and live inside the container.

Therefore:

- keypoints can persist
- fitted model data can persist
- the original uploaded video may disappear after restarting/replacing the container
- overlay rendering may then fail because the original source video is missing

A future improvement is to copy uploaded videos into a persistent directory such as:

```text
/app/results/videos/
```

which maps to:

```text
~/projects/MonocularBiomechanics/results/videos/
```

and save that persistent path in the fitted `.npz`.

---

## 29. Persistent cache management

The JAX and CUDA caches are stored in named Docker volumes, so they survive container removal and recreation.

List volumes:

```bash
docker volume ls
```

Inspect a cache volume:

```bash
docker volume inspect monocular-jax-cache
docker volume inspect monocular-cuda-cache
```

Remove the caches only if you intentionally want a completely fresh first-run compilation:

```bash
docker volume rm monocular-jax-cache
docker volume rm monocular-cuda-cache
```

Do **not** routinely delete these volumes. Keeping them avoids repeating JAX/CUDA compilation work.

Important distinction:

```text
docker stop / docker start  -> keeps the same container and all caches
docker rm                   -> deletes the container, but named cache volumes remain
docker volume rm            -> deletes the persistent cache itself
docker build                -> rebuilds the image; named cache volumes still remain
```

---

## 30. Check disk usage

Docker images can become large because TensorFlow, JAX, CUDA libraries, and model dependencies are substantial.

Docker disk usage:

```bash
docker system df
```

Detailed:

```bash
docker system df -v
```

---

## 31. Remove unused Docker resources

Remove stopped containers:

```bash
docker container prune
```

Remove unused images:

```bash
docker image prune
```

Remove unused build cache:

```bash
docker builder prune
```

Aggressive cleanup of all unused Docker resources:

```bash
docker system prune -a
```

Be careful with the last command because it removes unused images and build cache and can make future builds much slower.

---

## 32. Normal development workflow

From WSL:

```bash
cd ~/projects/MonocularBiomechanics
code .
```

Make code changes.

Check Python syntax:

```bash
python3 -m py_compile main.py
```

If you only want to run the existing environment again and have not rebuilt the image:

```bash
docker start -a monocular-test
```

If you changed source code that was copied into the image, rebuild and recreate the container:

```bash
docker build -t monocular-biomechanics .

docker rm -f monocular-test 2>/dev/null || true

docker run \
    --name monocular-test \
    --gpus all \
    -p 7860:7860 \
    -e OUTPUT_DIR=/app/results \
    -v "$(pwd)/results:/app/results" \
    -v monocular-jax-cache:/app/.jax_cache \
    -v monocular-cuda-cache:/root/.cache/cuda \
    monocular-biomechanics
```

Open:

```text
http://localhost:7860
```

In another terminal:

```bash
watch -n 1 nvidia-smi
```

And optionally:

```bash
watch -n 1 'ls -lh results'
```

---

## 33. Fast restart sequence

If you changed code and rebuilt the image:

```bash
docker rm -f monocular-test 2>/dev/null || true

docker build -t monocular-biomechanics .

docker run \
    --name monocular-test \
    --gpus all \
    -p 7860:7860 \
    -e OUTPUT_DIR=/app/results \
    -v "$(pwd)/results:/app/results" \
    -v monocular-jax-cache:/app/.jax_cache \
    -v monocular-cuda-cache:/root/.cache/cuda \
    monocular-biomechanics
```

---

## 34. Useful aliases (optional)

Add these to `~/.bashrc` if desired:

```bash
alias mb-build='docker build -t monocular-biomechanics .'
alias mb-stop='docker stop monocular-test'
alias mb-kill='docker rm -f monocular-test'
alias mb-logs='docker logs --tail 200 -f monocular-test'
alias mb-gpu='watch -n 1 nvidia-smi'
```

For the run command, a shell function is easier:

```bash
mb-run() {
    docker rm -f monocular-test 2>/dev/null || true
    docker run \
        --name monocular-test \
        --gpus all \
        -p 7860:7860 \
        -e OUTPUT_DIR=/app/results \
        -v "$(pwd)/results:/app/results" \
        monocular-biomechanics
}
```

Reload shell configuration:

```bash
source ~/.bashrc
```

Then:

```bash
mb-build
mb-run
```

---

## 35. Git workflow

Check changes:

```bash
git status
```

Stage Docker-related files:

```bash
git add Dockerfile .dockerignore main.py
```

Commit:

```bash
git commit -m "Add Docker GPU environment and persistent output handling"
```

Push branch:

```bash
git push -u origin docker-support
```

---

## 36. Quick command reference

### Build

```bash
docker build -t monocular-biomechanics .
```

### Run

```bash
docker run \
    --name monocular-test \
    --gpus all \
    -p 7860:7860 \
    -e OUTPUT_DIR=/app/results \
    -v "$(pwd)/results:/app/results" \
    -v monocular-jax-cache:/app/.jax_cache \
    -v monocular-cuda-cache:/root/.cache/cuda \
    monocular-biomechanics
```

### Open app

```text
http://localhost:7860
```

### Stop

```bash
docker stop monocular-test
```

### Kill/remove

```bash
docker rm -f monocular-test
```

### Logs

```bash
docker logs --tail 200 -f monocular-test
```

### Inspect exit reason

```bash
docker inspect monocular-test \
  --format='Status={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} Error={{.State.Error}}'
```

### Shell inside container

```bash
docker exec -it monocular-test bash
```

### GPU monitor

```bash
watch -n 1 nvidia-smi
```

### Results

```bash
ls -lh results
```

### Python syntax check

```bash
python3 -m py_compile main.py
```

### Restart WSL completely

Run in Windows PowerShell:

```powershell
wsl --shutdown
```

---

## 36. Current known caveats

1. **TensorFlow + RTX 5070 / Blackwell**
   - TensorFlow detects the GPU.
   - The current TensorFlow wheel may JIT compile PTX for compute capability `12.0a`.
   - First-run startup can therefore be unusually slow.

2. **MuJoCo / EGL rendering**
   - Fitting and GPU inference can work even if overlay/rendering later exposes WSL/EGL-specific problems.
   - Treat overlay generation as a separate compatibility test.

3. **Warp warnings**
   - Messages such as `Failed to import warp` are not automatically fatal.
   - Do not install Warp unless the actual code path requires it.

4. **Temporary uploaded videos**
   - Fitted model files can persist through `/app/results`.
   - Original uploaded videos still need persistent copying if overlays must work after a container restart.

5. **Do not use `--rm` for debugging runs**
   - It deletes the container immediately when the process exits.
   - Without the container, `docker logs` and `docker inspect` cannot be used afterward.
   - Use a named container such as `monocular-test` while debugging.

---

## 37. Recommended next improvement

Add persistent video storage:

```text
results/
├── videos/
├── *_keypoints.npz
├── *_fitted_model.npz
└── rendered/
```

Then copy uploaded Gradio videos into `results/videos/` before processing and save the persistent path in the fitted model file. This will make keypoints, fitted biomechanics, and overlay visualization reproducible across container restarts.
