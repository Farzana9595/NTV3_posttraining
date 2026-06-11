# Running `run_s3_pipeline.py` on a SageMaker AI instance

A reusable, copy-paste runbook. Works on any fresh SageMaker AI notebook /
Code Editor instance that already has `/opt/conda` (the default image does).

---

## 0. One-time setup on a NEW instance

```bash
cd /home/sagemaker-user/sagemaker-code-editor-server-data/script/NTV3_posttraining/rna-seq
bash setup_env.sh
```

That script is **idempotent** — it:

1. Sources conda into the current shell.
2. Creates the `ntv3-rnaseq` conda env (Python 3.12 + `tmux`) if missing.
3. Installs everything in `requirements.txt` (`boto3`, `requests`, `openpyxl`,
   `pybigtools`).
4. Prints the installed versions so you know it worked.

If you ever want to start over from scratch:

```bash
conda env remove -n ntv3-rnaseq -y
bash setup_env.sh
```

---

## 1. Verify AWS credentials

SageMaker normally gives you an attached IAM role automatically. Confirm:

```bash
aws sts get-caller-identity
```

You should see an ARN ending in `:assumed-role/...sagemaker_role/SageMaker`.
If not, fix credentials before running anything that talks to S3.

---

## 2. Run the pipeline inside `tmux`

`tmux` keeps the job alive when you close the browser tab / lose network.

### 2a. Start a session and run interactively

```bash
tmux new -s rnaseq
```

You are now inside tmux. Activate the env and launch:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate ntv3-rnaseq
cd /home/sagemaker-user/sagemaker-code-editor-server-data/script/NTV3_posttraining/rna-seq

# pick ONE mode (the script requires exactly one):
python run_s3_pipeline.py --status                            # free, just inspects S3 + local
python run_s3_pipeline.py --smoke-test 5     2>&1 | tee run_smoke.log
python run_s3_pipeline.py --full-rna-seq --resume 2>&1 | tee run_full.log
python run_s3_pipeline.py --download-all-bigwig 2>&1 | tee run_download.log
python run_s3_pipeline.py --pull-only
python run_s3_pipeline.py --push-only
```

Then **detach** (job keeps running):

```
Ctrl-b   then   d
```

### 2b. Start detached in one command (no interactive step)

```bash
tmux new -d -s rnaseq "source /opt/conda/etc/profile.d/conda.sh \
  && conda activate ntv3-rnaseq \
  && cd /home/sagemaker-user/sagemaker-code-editor-server-data/script/NTV3_posttraining/rna-seq \
  && python run_s3_pipeline.py --full-rna-seq --resume 2>&1 | tee run_full.log"
```

---

## 3. tmux cheat sheet

| Action                          | Command / keys                  |
| ------------------------------- | ------------------------------- |
| List sessions                   | `tmux ls`                       |
| Reattach to `rnaseq`            | `tmux attach -t rnaseq`         |
| Detach (job keeps running)      | `Ctrl-b`, then `d`              |
| Scroll up through output        | `Ctrl-b`, then `[` (q to exit)  |
| Kill the session                | `tmux kill-session -t rnaseq`   |
| Tail the log from outside tmux  | `tail -f rna-seq/run_full.log`  |

---

## 4. Pipeline modes (reference)

`run_s3_pipeline.py` requires **exactly one** mode flag:

| Flag                  | What it does                                                                 |
| --------------------- | ---------------------------------------------------------------------------- |
| `--status`            | Show S3 vs. local state. No downloads, no uploads.                           |
| `--smoke-test N`      | Matched pipeline on `N` samples. Use this first.                             |
| `--full`              | Full matched pipeline (small subset, strict scoring).                        |
| `--full-rna-seq`      | FULL pipeline over all ~1344 BigWigs → audit → references → prepare → S3.    |
| `--download-all-bigwig` | Just download all RNA-seq BigWigs from CyVerse (~126 GB).                  |
| `--pull-only`         | Only sync S3 → local.                                                        |
| `--push-only`         | Only sync local → S3.                                                        |

Useful add-ons: `--resume`, `--skip-pull`, `--skip-push`, `--dry-run`,
`--dry-run-downloads`, `--include-md5`.

---

## 5. Troubleshooting

- **`conda: command not found`** — run `source /opt/conda/etc/profile.d/conda.sh` first.
- **`tmux: command not found`** — you forgot to `conda activate ntv3-rnaseq`; tmux
  was installed inside that env (not system-wide).
- **`ModuleNotFoundError: pybigtools` (etc.)** — same: activate the env.
- **`NoCredentialsError` / `AccessDenied`** — check `aws sts get-caller-identity`
  and that the instance role has access to the target S3 bucket.
- **Disk full during `--full-rna-seq`** — the run can pull ~126 GB. Make sure
  `--data-root` points to a volume with enough space (default is on EFS under
  `NTV3_posttraining/data/instadepp_rna_seq`).
- **Resuming after disconnect** — your tmux session is still alive. Just
  `tmux attach -t rnaseq`. If the script itself died, re-run it with `--resume`.
