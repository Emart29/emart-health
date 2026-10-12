# N-ATLaS Health QA

A health-sector LoRA adapter for [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), with a 500-question
benchmark against the base model. Entry for the National AI Innovation Challenge 2026, Problem Statement 3
(Sectoral Fine-Tuning), health sector.

- **Model:** <https://huggingface.co/Emar7/N-ATLaS-Health-QA-LoRA-Powered-by-Awarri>
- **Base model:** `NCAIR1/N-ATLaS`. Every answer comes from N-ATLaS. No other language model is used.
- **Data:** [AfriMed-QA v2](https://huggingface.co/datasets/intronhealth/afrimedqa_v2), African medical
  multiple-choice questions.

## Result

Both models were scored on the same 500 held-out questions.

| Model | Correct | Accuracy | 95% CI (Wilson) |
| --- | --- | --- | --- |
| Base N-ATLaS | 252 / 500 | 50.4% | 46.0% to 54.8% |
| N-ATLaS + this adapter | 303 / 500 | 60.6% | 56.2% to 64.8% |
| Chance | | 21.2% | |

The adapter is 10.2 percentage points more accurate than base N-ATLaS. On the questions where the two models
disagree, base was right on 30 and the adapter on 81 (exact McNemar test, two-sided p = 1.4e-06).

This is one training run with one seed. Nothing was tuned on the 500 benchmark questions.

## What is in this repo

| File | Purpose |
| --- | --- |
| `natlas_pipeline.py` | The whole pipeline as one script: data prep, base scoring, QLoRA fine-tune, re-scoring, statistics, model card, publish. |
| `natlas_health_kaggle.ipynb` | The same code as a notebook, generated from the script. The reported run used this on Kaggle. |
| `natlas_health_demo.ipynb` | Live demo: loads N-ATLaS, then the published adapter on top, and answers the same questions with each. Its answers to the first five benchmark questions matched the published predictions 5 out of 5 for both models. |
| `modal_natlas_health.py` | Optional launcher that runs the script on a Modal GPU. Not used for the reported run. |
| `natlas_health_out/benchmark_summary.json` | Both accuracies, intervals, split method and settings, as written by the run. |
| `natlas_health_out/benchmark_500_predictions.csv` | One row per benchmark question with both models' answers. |
| `natlas_health_out/benchmark_by_specialty.csv` | Accuracy by medical specialty. |
| `natlas_health_out/benchmark_500_ids.csv` | The IDs of the 500 held-out questions. |
| `submission/` | The submission PDFs and the script that builds them from the result files. |

## How it works

1. Load AfriMed-QA v2 and keep multiple-choice questions with exactly one correct answer (3,722 of 4,039).
2. Hold out 500 questions at random with seed 42. The other 3,222 are the training set. The script stops if
   any benchmark question also appears in the training set.
3. Load N-ATLaS in 4-bit and score it on the 500 questions. The answer is the option letter with the highest
   next-token probability, so scoring is deterministic.
4. Fine-tune with QLoRA: rank 16 on all attention and MLP projection layers, 1 epoch, learning rate 2e-4.
5. Score the fine-tuned model on the same 500 questions with the same prompts.
6. Publish the adapter and the per-question results.

## Reproduce the run

You need a Hugging Face account that has accepted the terms for `NCAIR1/N-ATLaS` and
`intronhealth/afrimedqa_v2`, and a write token.

**On Kaggle (free, and how the reported run was produced):**

1. Import `natlas_health_kaggle.ipynb` as a new notebook.
2. Session options: Accelerator = GPU T4 x2, Internet = On.
3. Add-ons > Secrets: add `HF_TOKEN` and attach it to the notebook.
4. In the first cell, set `HF_USERNAME` to your own Hugging Face username.
5. Save Version > Save & Run All. The run takes about 51 minutes.

**On any Linux machine with a 16 GB or larger NVIDIA GPU:**

```bash
pip install "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" "accelerate>=0.33" datasets scipy
export HF_TOKEN=...            # write token with access to both gated repos
export HF_USERNAME=...         # where the adapter is published
export BATCH=2 GRAD_ACCUM=8    # 16 GB GPU. Use BATCH=4 GRAD_ACCUM=4 on 24 GB.
python natlas_pipeline.py      # set PUSH_TO_HUB=0 to skip publishing
```

The seed, the split and the scoring are fixed, so the base score reproduces exactly (it did, on two runs).
The fine-tuned score may shift slightly on a different GPU setup or library version. The reported run used
two T4 GPUs, which made the effective batch size 32 (101 optimizer steps).

One line of `natlas_pipeline.py` changed after the reported run: the p-value is now written with three
significant figures. The run itself rounded it to five decimal places, which is why
`benchmark_summary.json` shows `0.0`. The value 1.4e-06 is recomputed from the two disagreement counts in
that same file.

## Use the model

See the model card on Hugging Face for a loading example. The adapter loads on top of `NCAIR1/N-ATLaS` with
`peft.PeftModel.from_pretrained`.

## Limits

- Research and education only. This is not a medical device and must not be used for diagnosis or treatment.
- Trained and tested in English only. Nothing was evaluated in Yoruba, Hausa or Igbo.
- Single-answer multiple-choice questions only. Questions with more than one correct answer were excluded.
- The benchmark measures exam-style knowledge, not the quality or safety of free-text medical advice.
- The training data includes questions from the official AfriMed-QA test split, so this adapter must not be
  compared with published AfriMed-QA results. The only fair comparison is on the 500 held-out questions.
- The 500 questions come from the same dataset as the training questions. Exact duplicates were removed,
  but near-duplicates were not checked for.

## Licence and attribution

The adapter is a derivative of N-ATLaS and is released under the same N-ATLaS Terms of Use (research and
education use, capped at 1,000 active users).

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and
powered by Awarri Technologies.

Training and test data: AfriMed-QA v2 (CC BY-SA 4.0), Olatunji et al., 2024, arXiv:2411.15640.

## Author

Emmanuel Nwanguma, independent developer, Lagos. GitHub: [Emart29](https://github.com/Emart29).
