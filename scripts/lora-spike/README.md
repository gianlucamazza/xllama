# Deprecated shim

The host LoRA exploration pipeline moved to the **training pillar**:

→ [`training/README.md`](../../training/README.md)

```bash
./training/host/run_job.sh training/jobs/smollm2-360m-marker.json
# or
./build/linux-release/bin/xllama-cli --train-job training/jobs/smollm2-360m-marker.json
```

`run_spike.sh` still forwards to the training job for one release.
