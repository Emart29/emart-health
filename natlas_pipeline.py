# -*- coding: utf-8 -*-

# Generated from natlas_health_finetune_benchmark.ipynb. Same code, top to bottom.

# Needs HF_TOKEN in the environment. Optional env: HF_USERNAME, OUT_DIR, EPOCHS, BATCH, GRAD_ACCUM, PUSH_TO_HUB.


# ======================================================================
# N-ATLaS Health QA: fine-tune and 500-question benchmark
# ======================================================================


# ======================================================================
# 1. Config
# ======================================================================

import os, random
import numpy as np

BASE_MODEL = "NCAIR1/N-ATLaS"
DATASET = "intronhealth/afrimedqa_v2"

HF_USERNAME = os.environ.get("HF_USERNAME", "YOUR_HF_USERNAME")   # <-- change this (or set the env var)
ADAPTER_REPO = f"{HF_USERNAME}/N-ATLaS-Health-QA-LoRA-Powered-by-Awarri"
PUSH_TO_HUB = os.environ.get("PUSH_TO_HUB", "1") == "1"

SEED = 42
N_TEST = 500            # held-out benchmark size (the challenge asks for 500+)
MAX_PROMPT_TOKENS = 900 # questions longer than this are left out of both train and test
MAX_TRAIN_TOKENS = 640  # prompt + answer + rationale, per training example
RATIONALE_TOKENS = 160  # how much of the expert rationale to train on
USE_RATIONALE = True

EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = 2e-4
BATCH = int(os.environ.get("BATCH", 2))            # 2 fits a 16 GB T4; use 4 on a 24 GB GPU
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", 8))  # keep BATCH * GRAD_ACCUM = 16
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05

OUT_DIR = os.environ.get("OUT_DIR", "natlas_health_out")
DATE_STRING = "11 Oct 2026"   # fixed so prompts are identical on every run

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)

from huggingface_hub import login

def get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    from getpass import getpass
    return getpass("Hugging Face token: ")

login(token=get_hf_token())


# ======================================================================
# 2. Load the tokenizer and the data
# ======================================================================

import torch
from transformers import AutoTokenizer

torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

LETTERS = "ABCDEFGH"
LETTER_IDS = []
for ch in LETTERS:
    ids = tok.encode(ch, add_special_tokens=False)
    assert len(ids) == 1, f"letter {ch} is not a single token: {ids}"
    LETTER_IDS.append(ids[0])
EOT_ID = tok.convert_tokens_to_ids("<|eot_id|>")
assert EOT_ID is not None and EOT_ID != tok.unk_token_id, "could not find <|eot_id|> in the tokenizer"
print("letter token ids:", LETTER_IDS, "| eot id:", EOT_ID)

import pandas as pd
from datasets import load_dataset

raw = load_dataset(DATASET)
df = pd.concat([raw[s].to_pandas() for s in raw.keys()], ignore_index=True)
print("rows:", len(df))
print("columns:", df.columns.tolist())
print(df["question_type"].value_counts(dropna=False))

mcq = df[df["question_type"].astype(str).str.lower().str.contains("mcq")].copy()
print("MCQ rows:", len(mcq))
print("\nExample raw row (check the answer_options and correct_answer formats):")
print(mcq.iloc[0][["question", "answer_options", "correct_answer"]].to_dict())

import ast, json, re

def parse_options(x):
    # Returns [(key, text), ...] in option order, or None if it cannot be parsed.
    d = x if isinstance(x, dict) else None
    if d is None:
        s = str(x)
        for fn in (json.loads, ast.literal_eval):
            try:
                d = fn(s)
                break
            except Exception:
                d = None
    if not isinstance(d, dict):
        return None
    def order(k):
        m = re.search(r"(\d+)", str(k))
        return int(m.group(1)) if m else 999
    items = []
    for k, v in sorted(d.items(), key=lambda kv: order(kv[0])):
        t = "" if v is None else str(v).strip()
        if t and t.lower() not in ("nan", "none", "n/a"):
            items.append((str(k), t))
    return items if 2 <= len(items) <= len(LETTERS) else None

def parse_answer(ans, items):
    # Returns the index of the single correct option, or None (multi-answer or unparseable).
    if ans is None:
        return None
    s = str(ans).strip()
    keys = [k.lower() for k, _ in items]
    if s.lower() in keys:
        return keys.index(s.lower())
    if len(s) == 1 and s.upper() in LETTERS[:len(items)]:
        return LETTERS.index(s.upper())
    texts = [t.lower() for _, t in items]
    if s.lower() in texts:
        return texts.index(s.lower())
    return None

SYSTEM = ("You are a careful medical assistant answering multiple-choice medical exam questions. "
          "Reply with the letter of the single best option.")

def build_user(question, items):
    opts = "\n".join(f"{LETTERS[i]}. {t}" for i, (_, t) in enumerate(items))
    return f"Question:\n{question.strip()}\n\nOptions:\n{opts}\n\nAnswer with the letter of the single best option."

def chat_prompt(user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, date_string=DATE_STRING)

def norm(q):
    return re.sub(r"\s+", " ", str(q).lower()).strip()

rows, dropped = [], {"options": 0, "answer": 0, "empty": 0, "too_long": 0, "duplicate": 0}
seen = set()
for _, r in mcq.iterrows():
    q = r.get("question_clean")
    if not isinstance(q, str) or not q.strip():
        q = r.get("question")
    if not isinstance(q, str) or not q.strip():
        dropped["empty"] += 1; continue
    items = parse_options(r.get("answer_options"))
    if items is None:
        dropped["options"] += 1; continue
    idx = parse_answer(r.get("correct_answer"), items)
    if idx is None:
        dropped["answer"] += 1; continue
    key = norm(q)
    if key in seen:
        dropped["duplicate"] += 1; continue
    prompt = chat_prompt(build_user(q, items))
    if len(tok(prompt, add_special_tokens=False)["input_ids"]) > MAX_PROMPT_TOKENS:
        dropped["too_long"] += 1; continue
    seen.add(key)
    rat = r.get("answer_rationale")
    rows.append({
        "id": str(r.get("sample_id", r.get("sample id", len(rows)))),
        "question_key": key,
        "prompt": prompt,
        "n_options": len(items),
        "answer_idx": idx,
        "answer_letter": LETTERS[idx],
        "rationale": rat.strip() if isinstance(rat, str) else "",
        "specialty": str(r.get("specialty", "")),
        "country": str(r.get("country", "")),
        "tier": str(r.get("tier", "")),
        "official_split": str(r.get("split", "")).strip().lower(),
    })

print("usable MCQs:", len(rows), "| dropped:", dropped)
assert len(rows) >= N_TEST + 500, (
    "Too few usable questions. Look at the example row printed above and adjust parse_options / parse_answer.")

rng = random.Random(SEED)
official_test = [r for r in rows if r["official_split"] == "test"]
official_train = [r for r in rows if r["official_split"] != "test"]
print("usable MCQs in the official test split:", len(official_test), "| official train split:", len(official_train))

if len(official_test) >= N_TEST and len(official_train) >= 500:
    rng.shuffle(official_test)
    test_rows = official_test[:N_TEST]
    train_rows = official_train
    SPLIT_METHOD = (f"{N_TEST} questions sampled (seed {SEED}) from the dataset's official test split; "
                    "training used only the official train split")
else:
    order = list(range(len(rows)))
    rng.shuffle(order)
    test_rows = [rows[i] for i in order[:N_TEST]]
    train_rows = [rows[i] for i in order[N_TEST:]]
    SPLIT_METHOD = (f"{N_TEST} questions held out at random (seed {SEED}) from all usable single-answer "
                    f"questions before training; the dataset's official splits were not used because they hold "
                    f"{len(official_train)} usable train and {len(official_test)} usable test questions, so the "
                    "remaining questions, most of them from the official test split, were used for training")
print("split method:", SPLIT_METHOD)

overlap = {r["question_key"] for r in test_rows} & {r["question_key"] for r in train_rows}
assert not overlap, "train/test leakage"
print("train:", len(train_rows), "| test:", len(test_rows))
print("chance accuracy on the test set:", round(float(np.mean([1 / r["n_options"] for r in test_rows])), 4))

pd.DataFrame(test_rows).drop(columns=["prompt"]).to_csv(f"{OUT_DIR}/benchmark_500_ids.csv", index=False)


# ======================================================================
# 3. Load base N-ATLaS (4-bit) and score it on the 500 questions
# ======================================================================

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
                                             device_map={"": 0}, torch_dtype=torch.float16)
model.config.pad_token_id = tok.pad_token_id
print("loaded. GPU memory (GB):", round(torch.cuda.memory_allocated() / 1e9, 2))

@torch.no_grad()
def predict(m, data, batch_size=8):
    # Picks the option letter with the highest next-token probability. One forward pass per question.
    m.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(m.device)
        logits = m(**enc).logits[:, -1, :].float()
        for j, r in enumerate(batch):
            ids = LETTER_IDS[:r["n_options"]]
            preds.append(int(torch.argmax(logits[j, ids]).item()))
        if (i // batch_size) % 10 == 0:
            print(f"  scored {min(i + batch_size, len(data))}/{len(data)}")
    tok.padding_side = "right"
    return preds

base_pred = predict(model, test_rows)
gold = [r["answer_idx"] for r in test_rows]
base_ok = np.array([p == g for p, g in zip(base_pred, gold)])
print("BASE N-ATLaS accuracy:", round(base_ok.mean(), 4), f"({base_ok.sum()}/{len(gold)})")


# ======================================================================
# 4. Fine-tune with QLoRA
# ======================================================================

from datasets import Dataset

def encode_train(r):
    prompt_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
    target_ids = [LETTER_IDS[r["answer_idx"]]]
    if USE_RATIONALE and r["rationale"]:
        rat_ids = tok("\n\nRationale: " + r["rationale"], add_special_tokens=False)["input_ids"][:RATIONALE_TOKENS]
        target_ids += rat_ids
    budget = MAX_TRAIN_TOKENS - len(prompt_ids) - 1
    if budget < 1:
        return None
    target_ids = target_ids[:budget] + [EOT_ID]
    return {"input_ids": prompt_ids + target_ids,
            "labels": [-100] * len(prompt_ids) + target_ids}   # loss only on the answer

train_feats = [f for f in (encode_train(r) for r in train_rows) if f is not None]
print("training examples:", len(train_feats),
      "| mean length:", int(np.mean([len(f["input_ids"]) for f in train_feats])))
train_ds = Dataset.from_list(train_feats)

class Collator:
    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(feats), n), dtype=torch.long)
        lab = torch.full((len(feats), n), -100, dtype=torch.long)
        for i, f in enumerate(feats):
            k = len(f["input_ids"])
            ids[i, :k] = torch.tensor(f["input_ids"], dtype=torch.long)
            att[i, :k] = 1
            lab[i, :k] = torch.tensor(f["labels"], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments

model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

import math
total_steps = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM)) * EPOCHS
warmup_steps = max(1, int(0.05 * total_steps))   # 5% warmup, as a step count (warmup_ratio is gone in transformers 5)
print("optimizer steps:", total_steps, "| warmup steps:", warmup_steps)

args = TrainingArguments(
    output_dir=f"{OUT_DIR}/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=Collator())
train_out = trainer.train()
print(train_out)

# Save the adapter straight away, so a crash in scoring or the demo cannot lose the training run.
ADAPTER_DIR = f"{OUT_DIR}/adapter"
model.save_pretrained(ADAPTER_DIR)
tok.save_pretrained(ADAPTER_DIR)


# ======================================================================
# 5. Score the fine-tuned model on the same 500 questions
# ======================================================================

model.config.use_cache = True
tuned_pred = predict(model, test_rows)
tuned_ok = np.array([p == g for p, g in zip(tuned_pred, gold)])
print("FINE-TUNED accuracy:", round(tuned_ok.mean(), 4), f"({tuned_ok.sum()}/{len(gold)})")

from scipy.stats import binomtest

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return round(c - h, 4), round(c + h, 4)

n = len(gold)
only_base = int((base_ok & ~tuned_ok).sum())    # base right, fine-tuned wrong
only_tuned = int((~base_ok & tuned_ok).sum())   # fine-tuned right, base wrong
p_value = binomtest(min(only_base, only_tuned), only_base + only_tuned, 0.5).pvalue if (only_base + only_tuned) else 1.0

summary = {
    "benchmark": "AfriMed-QA v2 multiple-choice (single correct answer), held-out",
    "split_method": SPLIT_METHOD,
    "n_questions": n,
    "chance_accuracy": round(float(np.mean([1 / r["n_options"] for r in test_rows])), 4),
    "base_accuracy": round(float(base_ok.mean()), 4),
    "base_ci95": wilson(int(base_ok.sum()), n),
    "finetuned_accuracy": round(float(tuned_ok.mean()), 4),
    "finetuned_ci95": wilson(int(tuned_ok.sum()), n),
    "difference_points": round(100 * float(tuned_ok.mean() - base_ok.mean()), 2),
    "base_right_finetuned_wrong": only_base,
    "finetuned_right_base_wrong": only_tuned,
    "mcnemar_exact_p": float(f"{p_value:.3g}"),   # 3 significant figures, so a small value is not shown as 0.0
    "train_examples": len(train_feats),
    "epochs": EPOCHS, "lr": LR, "lora_r": LORA_R, "seed": SEED,
    "base_model": BASE_MODEL, "dataset": DATASET,
    "scoring": "argmax of next-token logits over the option letters, 4-bit NF4 weights for both models",
}
print(json.dumps(summary, indent=2))
with open(f"{OUT_DIR}/benchmark_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

res = pd.DataFrame(test_rows).drop(columns=["prompt", "question_key"])
res["base_pred"] = [LETTERS[p] for p in base_pred]
res["finetuned_pred"] = [LETTERS[p] for p in tuned_pred]
res["base_correct"] = base_ok
res["finetuned_correct"] = tuned_ok
res.to_csv(f"{OUT_DIR}/benchmark_500_predictions.csv", index=False)

by_spec = (res.groupby("specialty")
              .agg(n=("id", "count"), base=("base_correct", "mean"), finetuned=("finetuned_correct", "mean"))
              .query("n >= 10").sort_values("n", ascending=False).round(3))
by_spec.to_csv(f"{OUT_DIR}/benchmark_by_specialty.csv")
print(by_spec.head(15))


# ======================================================================
# **How to read the result.** Report both numbers exactly as printed, with the confidence intervals.
# ======================================================================


# ======================================================================
# 6. Try it (use this cell for the demo video)
# ======================================================================

@torch.no_grad()
def ask(question, options, max_new_tokens=220):
    items = [(f"option{i + 1}", o) for i, o in enumerate(options)]
    prompt = chat_prompt(build_user(question, items))
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.1,
                         repetition_penalty=1.12, eos_token_id=[EOT_ID, tok.eos_token_id],
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

demo = test_rows[0]
print(demo["prompt"].split("<|start_header_id|>user<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip())
print("\nGold answer:", demo["answer_letter"])
try:
    print("\nModel says:\n", ask(
        "A 4-year-old child in a malaria-endemic area presents with fever, pallor and a palpable spleen. "
        "Which investigation should be done first?",
        ["Chest X-ray", "Blood film for malaria parasites", "Lumbar puncture", "Urine culture"]))
except Exception as e:   # the demo must not stop the adapter from being published
    print("demo generation failed:", repr(e))


# ======================================================================
# 7. Save and publish the adapter
# ======================================================================

card =f'''---
base_model: {BASE_MODEL}
library_name: peft
datasets:
- {DATASET}
language:
- en
tags:
- medical
- lora
- n-atlas
---

# N-ATLaS Health QA LoRA, Powered by Awarri

A QLoRA adapter for [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}), fine-tuned on African
medical multiple-choice questions from AfriMed-QA. Built for the National AI Innovation Challenge 2026
(Problem Statement 3, Sectoral Fine-Tuning).

## Benchmark (500 held-out AfriMed-QA questions)

| Model | Accuracy | 95% CI |
|---|---|---|
| Base N-ATLaS | {summary["base_accuracy"]} | {summary["base_ci95"]} |
| This adapter | {summary["finetuned_accuracy"]} | {summary["finetuned_ci95"]} |

Chance accuracy: {summary["chance_accuracy"]}. Exact McNemar p-value: {summary["mcnemar_exact_p"]}.
Both models were scored with 4-bit weights by taking the most likely option letter.
Split: {SPLIT_METHOD}. No benchmark question appears in the training data.
Per-question predictions are in `benchmark_500_predictions.csv`.

## Limits

- Research and education only. This is not a medical device and must not be used for diagnosis or treatment.
- Trained and tested in English. It was not evaluated in Yoruba, Hausa or Igbo.
- The training data includes questions from the official AfriMed-QA test split. Do not compare this adapter
  with published AfriMed-QA results; the only fair comparison is on the 500 held-out questions listed in
  `benchmark_500_ids.csv`.
- Single-answer multiple-choice questions only. Questions with more than one correct answer were excluded.
- Trained for {EPOCHS} epoch on {len(train_feats)} questions.

## Licence and attribution

Released under the same N-ATLaS Terms of Use as the base model.
N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy,
and powered by Awarri Technologies.

Training and test data: AfriMed-QA v2 (CC BY-SA 4.0), Olatunji et al., 2024, arXiv:2411.15640.
'''
with open(f"{ADAPTER_DIR}/README.md", "w", encoding="utf-8") as f:
    f.write(card)

import shutil
for name in ("benchmark_summary.json", "benchmark_500_predictions.csv", "benchmark_by_specialty.csv", "benchmark_500_ids.csv"):
    shutil.copy(f"{OUT_DIR}/{name}", f"{ADAPTER_DIR}/{name}")

if PUSH_TO_HUB:
    assert HF_USERNAME != "YOUR_HF_USERNAME", "set HF_USERNAME in the config cell first"
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(ADAPTER_REPO, exist_ok=True)
    api.upload_folder(folder_path=ADAPTER_DIR, repo_id=ADAPTER_REPO)
    print("published:", f"https://huggingface.co/{ADAPTER_REPO}")

archive = shutil.make_archive(OUT_DIR.rstrip("/") + "_results", "zip", OUT_DIR)
print("results archive for the submission PDFs:", archive)
