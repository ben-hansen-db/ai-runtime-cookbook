# AI Runtime CLI quickstart

This is the runnable example corresponding to the [AI Runtime CLI quickstart](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/quickstart). It uploads `main.py`, runs it on one H100 GPU, and prints a greeting and the GPU name.

`workload.yaml` uses the Databricks AI environment version 6, which includes PyTorch.

## Run

Replace `PROFILE` with your Databricks CLI profile name.

```bash
databricks air run -p PROFILE --file workload.yaml --watch
```

To bring your own stack, replace the `environment` block with Standard environment version 6 and explicit dependencies:

```yaml
environment:
  version: 6
  dependencies:
    - torch==2.11.0
```

CLI runs install these dependencies on every run, so this option adds PyTorch and CUDA wheel download time.
