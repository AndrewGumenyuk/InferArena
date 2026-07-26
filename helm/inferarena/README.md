# InferArena Helm Chart

Deploy InferArena experiments on Kubernetes.

## Prerequisites

- Kubernetes 1.25+
- Helm 3.0+
- A published or locally built `inferarena` image

## Install

```bash
helm install my-experiment ./helm/inferarena \
  --set experiment.config="$(cat examples/experiment.yaml)"
```

## Run as a one-off Job

```bash
helm install my-experiment ./helm/inferarena \
  --set job.enabled=true \
  --set experiment.config="$(cat examples/experiment.yaml)"
```

## Use an existing ConfigMap

```bash
kubectl create configmap my-experiment-config --from-file=experiment.yaml=examples/experiment.yaml
helm install my-experiment ./helm/inferarena \
  --set experiment.existingConfigMap=my-experiment-config
```

## Uninstall

```bash
helm uninstall my-experiment
```

## Configuration

See `values.yaml` for the full list of options.
