# -*- coding: utf-8 -*-
"""Runs natlas_pipeline.py on a Modal GPU. Keep this file in the same folder as natlas_pipeline.py.

One-time setup (run this yourself, in your own terminal):
    modal secret create huggingface HF_TOKEN=<your Hugging Face write token>

Start the run (it keeps going if you close the terminal):
    modal run --detach modal_natlas_health.py --hf-username <your Hugging Face username>

Watch it:
    modal app logs natlas-health-qa

Get the results when it finishes:
    modal volume get natlas-health-out natlas_health_out ./natlas_health_out
"""
import os
import subprocess

import modal

app = modal.App("natlas-health-qa")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers>=4.44", "peft>=0.12", "bitsandbytes>=0.43", "accelerate>=0.33",
    "datasets", "scipy", "huggingface_hub", "pandas", "numpy",
)
if modal.is_local():
    here = os.path.dirname(os.path.abspath(__file__))
    image = image.add_local_file(os.path.join(here, "natlas_pipeline.py"), "/root/natlas_pipeline.py")

vol = modal.Volume.from_name("natlas-health-out", create_if_missing=True)


GPU = os.environ.get("NATLAS_GPU", "A10G")   # set NATLAS_GPU=L4 or T4 locally to launch on another GPU
BIG_GPU = GPU.upper() != "T4"                # 24 GB cards take batch 4; a 16 GB T4 takes batch 2


@app.function(image=image, gpu=GPU, timeout=6 * 60 * 60,
              secrets=[modal.Secret.from_name("huggingface")], volumes={"/out": vol})
def run(hf_username: str, epochs: int = 1, push: bool = True, batch: int = 4, grad_accum: int = 4):
    env = dict(os.environ,
               HF_USERNAME=hf_username,
               EPOCHS=str(epochs),
               BATCH=str(batch), GRAD_ACCUM=str(grad_accum),   # keep batch * grad_accum = 16
               OUT_DIR="/out/natlas_health_out",
               HF_HOME="/out/hf_cache",            # keeps the 16 GB model download between runs
               PUSH_TO_HUB="1" if push else "0")
    try:
        subprocess.run(["python", "-u", "/root/natlas_pipeline.py"], check=True, env=env)
    finally:
        vol.commit()


@app.local_entrypoint()
def main(hf_username: str, epochs: int = 1, push: bool = True):
    run.remote(hf_username, epochs, push, 4 if BIG_GPU else 2, 4 if BIG_GPU else 8)
