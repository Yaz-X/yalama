# YALAMA

## **Production-grade Native C++ LLaMA Runtime**

Deterministic high-performance **LibTorch inference runtime** built entirely in C++.

- No Python
- No Rust
- No HuggingFace runtime
- No wrappers

## Key Characteristics

- Native **C++ LibTorch runtime** (CUDA FP16)
- Direct **HuggingFace SafeTensors loading**
- Native **HF-compatible tokenizer** (byte-level BPE)
- Fixed-size **KV Cache**
- Optional KV Cache (can be disabled)
- **Greedy decoding**
- **TopK + Temperature sampling**
- Repetition mitigation (for base models)
- Soft stop sequences
- Deterministic execution
- **OpenAI-compatible API service**
- **REPL interactive Mode**

## Tested Models

- LLaMA 3.1 3B Instruct
- LLaMA 3.1 8B Base
- LLaMA 3.1 8B Instruct
- LLaMA 3.2 3B Instruct

Larger models (e.g. **70B**) are expected to work but were not tested due to hardware limits.

## Performance

**FP16 only** (no quantization).

Tested on **RTX 4090**  
KV Cache enabled, CUDA Graph disabled.

|      Decoding      |     Throughput    |
|--------------------|-------------------|
| No Sampling        | ~75–80 tokens/sec |
| Greedy             | ~55–65 tokens/sec |
| TopK + Temperature | ~50–60 tokens/sec |

## Model Format

YALAMA loads **HuggingFace SafeTensors models directly**.

Models must be downloaded separately from HuggingFace.

Supported architectures:

- LLaMA 3.x Base
- LLaMA 3.x Instruct

The runtime expects a HuggingFace model repository structure containing:

- `blobs/`
- `refs/`
- `snapshots/`

## Requirements

- NVIDIA GPU
- CUDA 12.8
- GCC 12+
- LibTorch (CUDA 12.8 build)
- CMake ≥ 3.20

## Docker Requirements

### Nvidia Container Toolkit

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   ca-certificates \
   curl \
   gnupg2

curl -fsSL <https://nvidia.github.io/libnvidia-container/gpgkey> | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L <https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list> | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.2-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

## Native Build

Set LibTorch path:

```bash
export LIBTORCH_PATH=/path_to_libtorch
```

Build YALAMA:

```bash
cmake -B build -G Ninja
cmake --build build
```

## Run

Default mode: **Service (OpenAI-compatible API)**

```bash
./build/yalama --model <path_to_model>
```

Example:

```bash
./build/yalama --model ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct
```

## Modes

YALAMA supports two runtime modes.

### Service Mode (default)

Runs the **OpenAI-compatible API server**.

```bash
--serviceMode 1
```

### REPL Mode

Runs an **interactive terminal session**.

```bash
--serviceMode 0
```

## CLI Arguments

`--model` is the **only required argument**.

|           Argument            |                              Description                                    |
|-------------------------------|-----------------------------------------------------------------------------|
| `--model`                     | **Required**. Path to HuggingFace SafeTensors model                         |
| `--config`                    | Folder containing `yalama_config.json`                                      |
| `--logs`                      | Log directory (used when debug is enabled). Default: current folder         |
| `--serviceMode`               | `1 = Service`, `0 = REPL`                                                   |
| `--httpThreadsPoolSize`       | `valid values from 4 to 64, only used in services mode`                     |
| `--port`                      | Service port (default `5067`)                                               |
| `--isServiceLoggingEnabled`   | Prints logging messages on terminal for services (default `false`)          |
| `--debug`                     |  `1 = enable`, `0 = disable`. Logs tensors per layer (performance impact)   |
| `--istorchvalidationsenabled` | `1 = enable`, `0 = disable`. Enables internal torch validations             |
| `--iskvcacheenabled`          | `1 = enable`, `0 = disable`. Default enabled                                |
| `--kvcachesizeingb`           | Minimum `1GB`, default `2GB`                                                |
| `--isgreedy`                  | `1 = enable`, `0 = disable`. Default enabled                                |
| `--topk`                      | `2–40`, default `40`. Ignored if greedy is enabled                          |
| `--temp`                      | `0.05–0.7`, default `0.6`. Ignored if greedy is enabled                     |
| `--showloadedweights`         | `1 = enable`, `0 = disable`. Shows weight names during loading              |

**NOTE: logs arg is path of the folder where the log file is written, the log file is huge and logs tensors in each step in each layer, and have a high performance
impace.**

## Config System

### Resolution Order

1. **CLI arguments** override everything
2. `yalama_config.json` fills missing values
3. **Defaults** applied last

Important:

- `--config` expects a **folder path**
- `--logs` expects a **folder path** to place the log file while running
- File name must be **`yalama_config.json`**
- Missing config file → defaults automatically applied

### Example `yalama_config.json`

```json
{
  "model": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct",
  "logs":"~/yalama_logs",
  "serviceMode":1,
  "httpThreadsPoolSize":32,
  "showloadedweights" : false,
  "port":"5067",
  "debug": false,
  "isServiceLoggingEnabled": false,
  "isTorchValidationsEnabled": false,
  "isKVCacheEnabled": true,
  "kvCacheSizeInGB": 2,
  "isGreedy": false,
  "topk": 40,
  "temp": 0.6
}
```

You can provide **only the fields you want to override**.  
Missing fields fall back to defaults.

## Default Runtime Settings

If no configuration is provided, the following defaults are used:

| Setting                     | Default Value | Notes                                                                             |
|-----------------------------|---------------|-----------------------------------------------------------------------------------|
| `debug`                     | `false`       | Debug logging disabled                                                            |
| `logs`                      | `./`          | Current directory, only active when debug is true                                 |
| `serviceMode`               | `true`        | OpenAI Compatible Service                                                         |
| `httpThreadsPoolSize`       | `32`          | Service Thread Pool size (ignored in REPL mode)                                   |
| `showloadedweights`         | `false`       | Show the names of loaded weights while loading                                    |
| `port`                      | `5067`        | OpenAI Compatible Service server port (ignored in REPL mode)                      |
| `isServiceLoggingEnabled`   | `false`       | print logs on terminal for service operations                                     |
|                             |               | status codes(i.e incoming request and response)                                   |
| `isTorchValidationsEnabled` | `false`       | Torch validations disabled                                                        |
| `isKVCacheEnabled`          | `true`        | KV cache enabled                                                                  |
| `kvCacheSizeInGB`           | `2`           | Minimum allowed: 1GB                                                              |
| `isGreedy`                  | `true`        | Greedy decoding enabled                                                           |
| `topk`                      | `40`          | Ignored if greedy enabled                                                         |
| `temp`                      | `0.6`         | Ignored if greedy enabled                                                         |

## Docker

### Prebuilt Docker Images

You can run YALAMA without building the image.

Pull the ready runtime image:

```bash
docker pull ghcr.io/yaz-x/yalama:latest
```

Run:

```bash
docker run -it --gpus all \
-p 5067:5067 \
-v ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct:/model \
ghcr.io/yaz-x/yalama:latest /app/yalama --model /model
```

or you can build images from source described in the next section as YALAMA provides Dockerfiles for both **native Linux** and **WSL environments**.

### Build Image

#### WSL / Linux

```bash
docker build --no-cache --rm -f docker/build/Dockerfile -t yalama .

docker image prune -f
```

---

## Run with Docker

### Minimal Run (OpenAI-compatible Service)

```bash
docker run -it --gpus all \
-p 5067:5067 \
-v ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct:/model \
yalama:latest /app/yalama --model /model
```

---

### Full Run (Config + Logs + Custom Port)

```bash
docker run -it --gpus all \
-p 8080:8080 \
-v ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct:/model \
-v ~/yalama_config:/config \
-v ~/yalama_logs:/logs \
yalama:linux \
--model /model \
--serviceMode 1 \
--port 8080 \
--httpThreadsPoolSize 32 \
--config /config \
--logs /logs \
--debug 0 \
--istorchvalidationsenabled 0 \
--isServiceLoggingEnabled 0 \
--iskvcacheenabled 1 \
--kvcachesizeingb 2 \
--isgreedy 1 \
--topk 40 \
--temp 0.6 \
--showloadedweights 0
```

Container supports the **same CLI arguments as native execution**.

### REPL Mode (Required args)

```bash
docker run -it --gpus all \
-v ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct:/models \
yalama:linux --model /models --serviceMode 0
```

## Native Execution

### OpenAI Service Mode

```bash
./build/yalama --model <path_to_model>
```

### REPL Mode (Native)

```bash
./build/yalama --model <path_to_model> --serviceMode 0
```

## OpenAI-Compatible Service API

### Health Check

```bash
curl http://localhost:5067/health
```

### Model Path

```bash
curl http://localhost:5067/model
```

### Chat Completion

```bash
curl http://localhost:5067/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    { "role": "user", "content": "Explain gravity in one sentence." }
  ]
}'
```

### Streaming Completion

```bash
curl http://localhost:5067/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "stream": true,
  "messages": [
    { "role": "user", "content": "Explain gravity in one sentence." }
  ]
}'
```

## Third-Party Libraries

| Library                     | License        | Link                                        |
|-----------------------------|----------------|-------------------------------------------  |
| NVIDIA CUDA Toolkit         | NVIDIA License | <https://developer.nvidia.com/cuda-toolkit> |
| LibTorch (PyTorch C++ API)  | BSD-style      | <https://pytorch.org>                       |
| cpp-httplib                 | MIT            | <https://github.com/yhirose/cpp-httplib>    |
| nlohmann/json               | MIT            | <https://github.com/nlohmann/json>          |

## License

**YALAMA Runtime**  
Copyright © 2026 Yazeed Hamdan

Licensed under the **Apache License, Version 2.0**.  
See the `LICENSE` file in the project root.

YALAMA **does not distribute model weights**.

LLaMA models are licensed separately under the **Meta Llama License**.  
Users are responsible for complying with the model license terms.
