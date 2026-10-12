---
base_model: NCAIR1/N-ATLaS
library_name: peft
datasets:
- intronhealth/afrimedqa_v2
language:
- en
tags:
- medical
- lora
- n-atlas
---

# N-ATLaS Health QA LoRA, Powered by Awarri

A QLoRA adapter for [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), fine-tuned on African
medical multiple-choice questions from AfriMed-QA. Built for the National AI Innovation Challenge 2026
(Problem Statement 3, Sectoral Fine-Tuning, health sector).

Code and full documentation: https://github.com/Emart29/emart-health

## Benchmark (500 held-out AfriMed-QA questions)

| Model | Correct | Accuracy | 95% CI (Wilson) |
|---|---|---|---|
| Base N-ATLaS | 252 / 500 | 50.4% | 46.0% to 54.8% |
| N-ATLaS + this adapter | 303 / 500 | 60.6% | 56.2% to 64.8% |
| Chance | | 21.2% | |

The adapter is 10.2 percentage points more accurate than base N-ATLaS on the same 500 questions.
Exact McNemar test: base right and adapter wrong on 30 questions, adapter right and base wrong on 81,
two-sided p = 1.4e-06. (`benchmark_summary.json` shows this p-value as 0.0 because the run rounded it
to five decimal places.)

Both models were scored with 4-bit weights by taking the most likely option letter.
Split: 500 questions held out at random (seed 42) from all usable single-answer questions before training; the dataset's official splits were not used because they hold 124 usable train and 3598 usable test questions, so the remaining questions, most of them from the official test split, were used for training. No benchmark question appears in the training data.
Per-question predictions are in `benchmark_500_predictions.csv`, and accuracy by specialty is in
`benchmark_by_specialty.csv`.

## How to use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "NCAIR1/N-ATLaS"   # gated: accept the terms on Hugging Face first
ADAPTER = "Emar7/N-ATLaS-Health-QA-LoRA-Powered-by-Awarri"

tok = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                            torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

system = ("You are a careful medical assistant answering multiple-choice medical exam questions. "
          "Reply with the letter of the single best option.")
user = ("Question:\nA 4-year-old child in a malaria-endemic area presents with fever, pallor and a "
        "palpable spleen. Which investigation should be done first?\n\nOptions:\nA. Chest X-ray\n"
        "B. Blood film for malaria parasites\nC. Lumbar puncture\nD. Urine culture\n\n"
        "Answer with the letter of the single best option.")
messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                 date_string="11 Oct 2026")
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=8, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))   # the training run printed: B
```

The adapter was trained to answer with an option letter, so it gives short answers to questions in this format.

## Training

QLoRA on 4-bit N-ATLaS: LoRA rank 16, alpha 32, dropout 0.05 on all attention and MLP projection layers
(41,943,040 trainable parameters, 0.52% of the model). 1 epoch on 3222 questions, learning rate 2e-4,
effective batch 32, 101 optimizer steps, seed 42. Trained on 2 x NVIDIA T4 on Kaggle in 44 minutes.
One run, no hyperparameter search, nothing tuned on the benchmark questions.

## Limits

- Research and education only. This is not a medical device and must not be used for diagnosis or treatment.
- Trained and tested in English. It was not evaluated in Yoruba, Hausa or Igbo.
- The training data includes questions from the official AfriMed-QA test split. Do not compare this adapter
  with published AfriMed-QA results; the only fair comparison is on the 500 held-out questions listed in
  `benchmark_500_ids.csv`.
- Single-answer multiple-choice questions only. Questions with more than one correct answer were excluded.
- The benchmark measures exam-style knowledge, not the quality or safety of free-text medical advice.
- Trained for 1 epoch on 3222 questions.

## Licence and attribution

Released under the same N-ATLaS Terms of Use as the base model.
N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy,
and powered by Awarri Technologies.

Training and test data: AfriMed-QA v2 (CC BY-SA 4.0), Olatunji et al., 2024, arXiv:2411.15640.
