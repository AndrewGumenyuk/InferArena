# Deployment

InferArena is distributed as a Python package and a Docker image. This guide covers how to run it reproducibly with Docker, optionally backed by a real vLLM server.

## Docker image

The multi-stage `Dockerfile` provides two targets:

- `base` — CLI-only image with `inferarena` installed plus plotting extras.
- `dev` — includes test dependencies and the full test suite.

### Build

```bash
docker build -t inferarena .
```

Use the development target for running tests inside a container:

```bash
docker build --target dev -t inferarena:dev .
```

### Run a built-in experiment

```bash
docker run --rm -it \
  -v $(pwd)/inferarena_outputs:/app/inferarena_outputs \
  inferarena run --config examples/experiment.yaml
```

### Run the comparison command

```bash
docker run --rm -it \
  -v $(pwd)/inferarena_outputs:/app/inferarena_outputs \
  inferarena compare --config examples/experiment.yaml \
  --schedulers fcfs,chunked_prefill
```

## Docker Compose

`docker-compose.yml` defines the `inferarena` service and an optional `vllm` profile for real-cluster experiments.

### Simulation only

```bash
docker compose run --rm inferarena run --config examples/experiment.yaml
```

### With vLLM backend

This requires an NVIDIA GPU with the Docker NVIDIA runtime installed.

```bash
# Start the vLLM server in the background.
docker compose --profile vllm up -d vllm

# Run a real-cluster experiment against it.
docker compose run --rm inferarena run --config examples/experiment_vllm.yaml

# Tear everything down.
docker compose --profile vllm down
```

The example config points to `http://vllm:8000/v1`, which resolves inside the Compose network.

## Kubernetes (experimental)

A minimal deployment would look like:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inferarena
spec:
  replicas: 1
  selector:
    matchLabels:
      app: inferarena
  template:
    metadata:
      labels:
        app: inferarena
    spec:
      containers:
        - name: inferarena
          image: inferarena:latest
          command: ["inferarena", "benchmark"]
          volumeMounts:
            - name: outputs
              mountPath: /app/inferarena_outputs
      volumes:
        - name: outputs
          emptyDir: {}
```

For production use, mount a persistent volume for `inferarena_outputs` and configure a sidecar or job runner to execute experiments on demand.

## Resource considerations

- **Simulation workloads** are CPU-only and cheap to run. Allocate 1–2 CPU cores and 1 GiB of RAM for modest experiments.
- **Real-cluster backends** such as vLLM require NVIDIA GPUs. Plan GPU memory based on the model size and `max-num-seqs` setting.
- **Large trace datasets** may need additional memory and disk for pre-processing; mount a volume for `~/.cache/inferarena` if you download public datasets.
