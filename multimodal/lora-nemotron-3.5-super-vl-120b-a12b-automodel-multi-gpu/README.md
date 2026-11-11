# LoRA fine-tuning of Nemotron 3.5 Super VL with NeMo AutoModel (multi-GPU)

This recipe uses NeMo AutoModel to fine-tune `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` on the CORD-v2 receipt-image dataset with LoRA. The model is a vision-language mixture of experts. Its vision and audio towers stay frozen, and multi-token prediction (MTP) is disabled.

Training runs for 50 steps on one 8x H100 node. It uses FSDP2, eight-way expert parallelism, rank-64 LoRA, and the GMM/DeepEP backend. The local batch size is 1 per GPU, for a global batch of 8 (no gradient accumulation). At the final step, it runs validation and saves a checkpoint. The recipe logs training metrics only. Measuring CORD-v2 field-extraction quality requires a separate evaluation.

Environment configurations: NVIDIA's stock `nemo-automodel:26.08.00` image does not yet support Super VL. To add it, the recipe builds a custom [AIR container image](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) from the bundled `Dockerfile`, which layers Super VL support on top of that image.

Workload execution takes about 12 minutes. For the first run, the one-time image build and upload may take additional time.

## Before you run

Complete these steps in order from this recipe's directory. On macOS, the image is built locally in Docker Desktop; training runs on AI Runtime.

1. Configure and authenticate the Databricks CLI using the [cookbook setup instructions](../../README.md#run-a-recipe). Ask a workspace admin to enable the previews required for custom images: [AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/docker-images) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/docker-images).

2. Install [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), choosing the installer for your Mac's chip (Apple silicon or Intel). Open Docker Desktop and wait until its engine is running. Keep it running during both the build and the image push. AIR requires a `linux/amd64` image, including when building on Apple silicon; keep `--platform linux/amd64` in the build command below. The image is several GB, so allow space for the base image, build cache, and final image in Docker Desktop's disk allocation.

3. Choose an existing Unity Catalog catalog and schema where you can push images. This example uses `main.default`; see the custom image guide above for the required Artifact Registry privileges. Check `env_variables.ADAPTER_OUTPUT_ROOT` in `workload.yaml`. Its default is:

   ```text
   /Volumes/main/default/air_examples/lora-nemotron-3.5-super-vl-120b-a12b-automodel-multi-gpu
   ```

   The volume must already exist and be writable. To use another location, update `ADAPTER_OUTPUT_ROOT` to a directory in another existing writable volume.

4. Check that the CLI can access AIR image commands and that Docker can reach its running engine before starting the large build:

   ```bash
   databricks air images push --help
   docker info
   ```

   The first command should show image-push usage. The second should show a `Server` section without a connection error. If `docker` is missing, finish the Docker Desktop installation; if it cannot connect, start Docker Desktop and wait for the engine to be ready.

5. Build the Super VL image for `linux/amd64` and push it into Unity Catalog:

   ```bash
   docker build --platform linux/amd64 -t nemo-automodel-vl:26.08.00 .
   databricks air images push -s nemo-automodel-vl:26.08.00 \
     --catalog main --schema default --artifact nemo-automodel-vl:26.08.00
   ```

   `--artifact` names the destination so it matches `environment.unity_catalog_image` in `workload.yaml`. This command creates `main.default.nemo-automodel-vl:26.08.00`. To use a different catalog or schema, change `--catalog`/`--schema` above and update `environment.unity_catalog_image` to match. The build and push can take a while.

The Dockerfile's base image `nvcr.io/nvidia/nemo-automodel:26.08.00` is public, and it fetches the AutoModel source from public GitHub. You therefore do not need an NGC account, API key, or Docker login to build it. If you switch to an NGC image that requires authentication, run `docker login nvcr.io` first. Use `$oauthtoken` as the username and your NGC API key as the password.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

AutoModel logs training loss, learning rate, gradient norm, and throughput every five steps to the AIR MLflow run. It logs validation loss at the final step. The final adapter and config are uploaded to `final_adapter` in the run, and copied to `ADAPTER_OUTPUT_ROOT/<MLFLOW_RUN_ID>/final_adapter` on the configured Unity Catalog volume. Checkpoints stay on the node's local disk.
