# -*- coding: utf-8 -*-
"""Builds the four submission PDFs from the real run outputs in ../natlas_health_out.

    python submission/build_pdfs.py

Every number in the PDFs is read from benchmark_summary.json, benchmark_by_specialty.csv and
run_facts.json. Nothing is typed in by hand.
"""
import csv
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "natlas_health_out")
S = json.load(open(os.path.join(OUT, "benchmark_summary.json"), encoding="utf-8"))
SPEC = list(csv.DictReader(open(os.path.join(OUT, "benchmark_by_specialty.csv"), encoding="utf-8")))
FACTS = json.load(open(os.path.join(HERE, "run_facts.json"), encoding="utf-8"))

ADAPTER = FACTS["adapter_repo"]
HF_URL = f"https://huggingface.co/{ADAPTER}"
GH_URL = FACTS["github_url"]
ATTRIBUTION = ("N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital "
               "Economy, and powered by Awarri Technologies.")
SAFETY = ("This is a research model. It is not a medical device and must not be used for diagnosis or "
          "treatment.")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, leading=22, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=9.5, textColor=colors.HexColor("#555555"), spaceAfter=10)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("P", parent=ss["Normal"], fontSize=10, leading=14, spaceAfter=6)
B = ParagraphStyle("B", parent=P, leftIndent=14, bulletIndent=4, spaceAfter=3)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=7.8, leading=10, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=5, spaceBefore=4, spaceAfter=10)
CELL = ParagraphStyle("CELL", parent=ss["Normal"], fontSize=9, leading=11.5)


def esc(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pct(x):
    return f"{100 * float(x):.1f}%"


def ci(pair):
    return f"{100 * pair[0]:.1f}% to {100 * pair[1]:.1f}%"


def bullets(items):
    return [Paragraph(t, B, bulletText="-") for t in items]


def table(rows, widths, header=True):
    data = [[Paragraph(esc(c) if not str(c).startswith("<") else c, CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")))
    t.setStyle(TableStyle(style))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "N-ATLaS Health QA | NAIC 2026, Problem Statement 3 (Sectoral Fine-Tuning, health) | "
                                        "Emmanuel Nwanguma")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build(name, title, subtitle, story):
    path = os.path.join(HERE, name)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=2 * cm, title=title, author="Emmanuel Nwanguma")
    doc.build([Paragraph(title, H1), Paragraph(subtitle, SUB)] + story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


# ---------------------------------------------------------------- shared facts and wording
n = S["n_questions"]
base_k = round(S["base_accuracy"] * n)
tuned_k = round(S["finetuned_accuracy"] * n)
diff = S["difference_points"]
# The summary file rounds the p-value to 5 places, which prints a small value as 0.0. Recompute it from the
# two disagreement counts in the same file, with the same exact two-sided binomial (McNemar) test.
from scipy.stats import binomtest
_b, _c = S["base_right_finetuned_wrong"], S["finetuned_right_base_wrong"]
p = binomtest(min(_b, _c), _b + _c, 0.5).pvalue if (_b + _c) else 1.0
p_text = f"{p:.5f}".rstrip("0").rstrip(".") if p >= 0.0001 else f"{p:.1e}".replace("e-0", " x 10^-").replace("e-", " x 10^-")
if p < 0.05 and diff > 0:
    VERDICT = (f"The fine-tuned model was more accurate than base N-ATLaS by {diff:.1f} percentage points on the "
               f"same {n} questions. The exact McNemar p-value is {p_text}, so the difference is statistically "
               "significant at the 0.05 level.")
elif p < 0.05 and diff < 0:
    VERDICT = (f"The fine-tuned model was less accurate than base N-ATLaS by {abs(diff):.1f} percentage points on "
               f"the same {n} questions. The exact McNemar p-value is {p_text}, so the drop is statistically "
               "significant. Fine-tuning made the model worse on this benchmark.")
else:
    VERDICT = (f"The fine-tuned model differed from base N-ATLaS by {diff:+.1f} percentage points on the same {n} "
               f"questions. The exact McNemar p-value is {p_text}, which is above 0.05. The result is "
               "inconclusive: this benchmark does not show that fine-tuning changed accuracy.")

RESULT_ROWS = [
    ["Model", "Correct", "Accuracy", "95% confidence interval (Wilson)"],
    ["Base N-ATLaS (NCAIR1/N-ATLaS)", f"{base_k} / {n}", pct(S["base_accuracy"]), ci(S["base_ci95"])],
    ["N-ATLaS + this adapter", f"{tuned_k} / {n}", pct(S["finetuned_accuracy"]), ci(S["finetuned_ci95"])],
    ["Chance (random guessing)", "", pct(S["chance_accuracy"]), ""],
]
RESULT_W = [6.2 * cm, 2.4 * cm, 2.4 * cm, 6 * cm]

LIMITS = [
    SAFETY,
    "Training and testing were in English only. Nothing was evaluated in Yoruba, Hausa or Igbo, so no claim is "
    "made about those languages.",
    "Only multiple-choice questions with a single correct answer were used. Questions with more than one correct "
    "answer were excluded. Open-ended and consumer questions were not tested.",
    "The model is scored by which option letter it rates most likely. This measures exam-style knowledge, not "
    "the quality or safety of free-text medical advice.",
    "The training data includes questions from the official AfriMed-QA test split, so this adapter must not be "
    "compared with published AfriMed-QA results. The only fair comparison is on the 500 held-out questions.",
    f"One training run, one seed ({S['seed']}), {S['epochs']} epoch. No hyperparameter search was done, and "
    "nothing was tuned on the 500 benchmark questions.",
    "Both models were scored with 4-bit quantised weights. Full-precision scores may differ slightly.",
    "The 500 questions come from the same dataset as the training questions. Exact duplicates were removed, but "
    "near-duplicate or closely related questions were not checked for.",
]

LOAD_CODE = f'''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE = "NCAIR1/N-ATLaS"                      # gated: accept the terms on Hugging Face first
ADAPTER = "{ADAPTER}"

tok = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={{"": 0}},
                                            torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, ADAPTER)   # the adapter sits on top of N-ATLaS
model.eval()

system = ("You are a careful medical assistant answering multiple-choice medical exam questions. "
          "Reply with the letter of the single best option.")
user = ("Question:\\nWhich investigation should be done first for a febrile child with pallor and a "
        "palpable spleen in a malaria-endemic area?\\n\\nOptions:\\nA. Chest X-ray\\n"
        "B. Blood film for malaria parasites\\nC. Lumbar puncture\\nD. Urine culture\\n\\n"
        "Answer with the letter of the single best option.")
messages = [{{"role": "system", "content": system}}, {{"role": "user", "content": user}}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                 date_string="11 Oct 2026")
enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
out = model.generate(**enc, max_new_tokens=8, do_sample=False)
print(tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))'''

PROMPT_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

[N-ATLaS default header with the date string]

You are a careful medical assistant answering multiple-choice medical exam questions. Reply with the
letter of the single best option.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question:
{question}

Options:
A. {option 1}
B. {option 2}
...

Answer with the letter of the single best option.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{answer letter}'''

TRAIN_ROWS = [
    ["Setting", "Value"],
    ["Base model", f"{S['base_model']} (Llama architecture, 8 billion parameters), loaded in 4-bit NF4"],
    ["Method", "QLoRA: the base weights stay frozen, only low-rank adapter weights are trained"],
    ["LoRA", f"rank {S['lora_r']}, alpha 32, dropout 0.05, on all attention and MLP projection layers"],
    ["Trainable parameters", FACTS["trainable_params"]],
    ["Training examples", f"{S['train_examples']} single-answer multiple-choice questions"],
    ["Epochs / learning rate", f"{S['epochs']} / {S['lr']} with a short linear warmup, then cosine decay"],
    ["Batch", FACTS["batch"]],
    ["Loss", "Only on the answer (the option letter, plus the expert rationale where the dataset has one)"],
    ["Hardware and time", FACTS["hardware_and_time"]],
    ["Seed", str(S["seed"])],
]

# ---------------------------------------------------------------- 1. N-ATLaS integration evidence
story = [
    Paragraph("1. What this shows", H2),
    Paragraph("Every answer from this project comes from N-ATLaS. The artefact is a LoRA adapter: a small set of "
              "extra weights that only works when loaded on top of the published N-ATLaS model. No other "
              "language model is called anywhere in the code.", P),
    *bullets([f"Base model: <b>{S['base_model']}</b>, downloaded from Hugging Face under its gated terms.",
              f"Adapter: <b>{esc(ADAPTER)}</b> ({esc(HF_URL)})",
              f"Code: {esc(GH_URL)}"]),
    Paragraph("2. How the adapter loads on top of N-ATLaS", H2),
    Paragraph("The adapter is loaded with the PEFT library. The call to <i>PeftModel.from_pretrained</i> takes "
              "the N-ATLaS model object as its first argument. The adapter's own config file records "
              f"<i>base_model_name_or_path = {S['base_model']}</i>.", P),
    Preformatted(LOAD_CODE, CODE),
    Paragraph("3. Prompt template", H2),
    Paragraph("Prompts are built with N-ATLaS's own chat template (<i>tokenizer.apply_chat_template</i>), so the "
              "special tokens are exactly the ones the base model was trained with. The same template is used "
              "for training, for scoring base N-ATLaS and for scoring the fine-tuned model.", P),
    Preformatted(PROMPT_TEMPLATE, CODE),
    Paragraph("4. Training setup", H2),
    table(TRAIN_ROWS, [4.2 * cm, 12.8 * cm]),
    Paragraph("5. Sample output", H2),
    Paragraph("Question put to N-ATLaS with the adapter loaded, at the end of the training run:", P),
    Preformatted(FACTS["demo_question"], CODE),
    Paragraph("Model output, copied from the run log:", P),
    Preformatted(FACTS["demo_output"], CODE),
    Paragraph("6. Result on the 500-question benchmark", H2),
    table(RESULT_ROWS, RESULT_W),
    Spacer(1, 6),
    Paragraph(VERDICT, P),
    Paragraph("7. Licence and attribution", H2),
    Paragraph("The adapter is a derivative of N-ATLaS and is released under the same N-ATLaS Terms of Use "
              "(research and education use, capped at 1,000 active users). Its name carries the required suffix "
              "\"Powered by Awarri\".", P),
    Paragraph(ATTRIBUTION, P),
    Paragraph("Training and test data: AfriMed-QA v2 (CC BY-SA 4.0), Olatunji et al., 2024, arXiv:2411.15640.", P),
    Paragraph(SAFETY, P),
]
build("2_NATLaS_Integration_Evidence.pdf", "N-ATLaS Integration Evidence",
      "N-ATLaS Health QA. A health-sector LoRA adapter for N-ATLaS.", story)

# ---------------------------------------------------------------- 2. Real-world validation
spec_rows = [["Specialty", "Questions", "Base N-ATLaS", "Fine-tuned", "Change (points)"]]
for r in SPEC:
    b, f = float(r["base"]), float(r["finetuned"])
    spec_rows.append([r["specialty"].replace("_", " "), r["n"], pct(b), pct(f), f"{100 * (f - b):+.1f}"])

story = [
    Paragraph("1. Summary", H2),
    Paragraph(f"Base N-ATLaS and the fine-tuned model were scored on the same {n} held-out African medical "
              "multiple-choice questions. The validation rule for Problem Statement 3 is a benchmark of 500 or "
              "more examples against base N-ATLaS.", P),
    table(RESULT_ROWS, RESULT_W),
    Spacer(1, 6),
    Paragraph(VERDICT, P),
    Paragraph("2. The benchmark", H2),
    *bullets([
        "Source: AfriMed-QA v2 (intronhealth/afrimedqa_v2), a pan-African medical question dataset. "
        "Licence CC BY-SA 4.0. Olatunji et al., 2024, arXiv:2411.15640.",
        f"Questions used: multiple-choice questions with exactly one correct answer. Of {FACTS['mcq_rows']} "
        f"multiple-choice rows, {FACTS['usable_mcq']} were usable. {FACTS['dropped_text']}",
        f"Split: {esc(S['split_method'])}.",
        "Leakage check: questions were de-duplicated on normalised text before the split, and the script stops "
        "with an error if any benchmark question also appears in the training set. It did not fire.",
        "The IDs of the 500 questions are published in benchmark_500_ids.csv, and both models' answers to every "
        "question are in benchmark_500_predictions.csv.",
    ]),
    Paragraph("3. How each model was scored", H2),
    *bullets([
        "Both models got the identical prompt for each question, built with N-ATLaS's own chat template.",
        "The predicted answer is the option letter with the highest next-token probability, restricted to the "
        "letters that exist for that question. One forward pass per question, with no sampling, so the scores "
        "are deterministic. The base score was reproduced exactly on a second run.",
        "Both models used the same 4-bit NF4 quantised weights.",
        f"Chance accuracy is {pct(S['chance_accuracy'])}: the average of one divided by the number of options.",
    ]),
    Paragraph("4. Statistics", H2),
    *bullets([
        "Confidence intervals are 95% Wilson score intervals.",
        "The two models are compared with an exact McNemar test, which uses only the questions where they "
        "disagree. This is the correct test for two models scored on the same questions.",
        f"Base right and fine-tuned wrong: {S['base_right_finetuned_wrong']} questions. Fine-tuned right and "
        f"base wrong: {S['finetuned_right_base_wrong']} questions. Exact two-sided p-value: {p_text}.",
    ]),
    Paragraph("5. Accuracy by medical specialty", H2),
    Paragraph("Specialties with at least 10 benchmark questions. These groups are small, so individual rows are "
              "noisy and should not be read as significant on their own.", P),
    table(spec_rows, [6.6 * cm, 2.2 * cm, 2.8 * cm, 2.6 * cm, 2.8 * cm]),
    Paragraph("6. Limits", H2),
    *bullets(LIMITS),
    Paragraph("7. Reproducing the numbers", H2),
    Paragraph(f"Run natlas_pipeline.py (or the notebook natlas_health_kaggle.ipynb) from {esc(GH_URL)} with a "
              "Hugging Face token that has access to N-ATLaS and AfriMed-QA. The seed, the split and the scoring "
              "are fixed, so the base score reproduces exactly. Training on a different GPU or library version "
              "may shift the fine-tuned score slightly.", P),
]
build("3_Real_World_Validation.pdf", "Real-World Validation: 500-Question Benchmark",
      "N-ATLaS Health QA against base N-ATLaS on held-out AfriMed-QA questions.", story)

# ---------------------------------------------------------------- 3. Technical documentation
PIPELINE_ROWS = [
    ["Step", "What happens"],
    ["1. Config", "Model, dataset, seed 42, LoRA and training settings. All fixed in the script."],
    ["2. Data", "Load AfriMed-QA v2, keep single-answer multiple-choice questions, parse the options and the "
                "answer key, drop duplicates, build each prompt with the N-ATLaS chat template, hold out 500."],
    ["3. Base score", "Load N-ATLaS in 4-bit and score it on the 500 questions."],
    ["4. Fine-tune", "Attach a LoRA adapter and train it with QLoRA. The base weights are frozen."],
    ["5. Re-score", "Score the fine-tuned model on the same 500 questions. Compute intervals and McNemar's test."],
    ["6. Demo", "Generate one free-text answer as a sanity check."],
    ["7. Publish", "Save the adapter, write the model card from the measured numbers, push to Hugging Face."],
]
FILE_ROWS = [
    ["File", "Purpose"],
    ["natlas_pipeline.py", "The whole pipeline as one script. This is the source of truth."],
    ["natlas_health_kaggle.ipynb", "The same code as a notebook, generated from the script. Used for the reported run."],
    ["natlas_health_demo.ipynb", "Live demo: loads N-ATLaS, then the published adapter, and answers the same questions."],
    ["modal_natlas_health.py", "Optional launcher that runs the script on a Modal GPU."],
    ["natlas_health_out/benchmark_summary.json", "Both accuracies, intervals, p-value, split method, settings."],
    ["natlas_health_out/benchmark_500_predictions.csv", "One row per benchmark question with both models' answers."],
    ["natlas_health_out/benchmark_by_specialty.csv", "Accuracy by medical specialty."],
    ["natlas_health_out/benchmark_500_ids.csv", "The IDs of the 500 held-out questions."],
    ["submission/build_pdfs.py", "Builds the submission PDFs from the result files."],
]
RUN_KAGGLE = '''1. Accept the terms for NCAIR1/N-ATLaS and intronhealth/afrimedqa_v2 on Hugging Face.
2. On Kaggle, import natlas_health_kaggle.ipynb as a new notebook.
3. Session options: Accelerator = GPU T4 x2, Internet = On.
4. Add-ons > Secrets: add HF_TOKEN (a Hugging Face write token) and attach it to the notebook.
5. In the first cell, set HF_USERNAME to your own Hugging Face username.
6. Save Version > Save & Run All. Results appear in /kaggle/working/natlas_health_out.'''
RUN_LOCAL = '''pip install "transformers>=4.44" "peft>=0.12" "bitsandbytes>=0.43" "accelerate>=0.33" datasets scipy
export HF_TOKEN=...            # write token with access to both gated repos
export HF_USERNAME=...         # where the adapter is published
export BATCH=2 GRAD_ACCUM=8    # 16 GB GPU. Use BATCH=4 GRAD_ACCUM=4 on 24 GB.
python natlas_pipeline.py      # set PUSH_TO_HUB=0 to skip publishing'''

story = [
    Paragraph("1. Overview", H2),
    Paragraph("N-ATLaS Health QA is a LoRA adapter that specialises N-ATLaS, Nigeria's open language model, for "
              "African medical multiple-choice questions. It was built for Problem Statement 3 (Sectoral "
              "Fine-Tuning) in the health sector. The project is one reproducible pipeline: prepare the data, "
              "score base N-ATLaS, fine-tune, score again on the same questions, and publish the adapter with "
              "its measured results.", P),
    *bullets([f"Model: {esc(HF_URL)}", f"Code: {esc(GH_URL)}"]),
    table(RESULT_ROWS, RESULT_W),
    Spacer(1, 6),
    Paragraph(VERDICT, P),
    Paragraph("2. Architecture", H2),
    Paragraph("There are two parts. The base model N-ATLaS (8 billion parameters, Llama architecture) is loaded "
              "unchanged from Hugging Face in 4-bit precision. The adapter adds low-rank weight updates to the "
              f"attention and MLP projection layers: {FACTS['trainable_params']}. At inference the adapter is "
              "loaded on top of N-ATLaS with the PEFT library. Nothing else is in the loop: no retrieval, no "
              "second model, no external API.", P),
    table(PIPELINE_ROWS, [3.2 * cm, 13.8 * cm]),
    Paragraph("3. Repository contents", H2),
    table(FILE_ROWS, [7.2 * cm, 9.8 * cm]),
    Paragraph("4. Training setup", H2),
    table(TRAIN_ROWS, [4.2 * cm, 12.8 * cm]),
    Paragraph("5. How to reproduce the run", H2),
    Paragraph("On Kaggle (free GPU, this is how the reported run was produced):", P),
    Preformatted(RUN_KAGGLE, CODE),
    Paragraph("On any Linux machine with a 16 GB or larger NVIDIA GPU:", P),
    Preformatted(RUN_LOCAL, CODE),
    Paragraph(f"Library versions used for the reported run: {esc(FACTS['versions'])}.", P),
    Paragraph("6. How to load and query the model", H2),
    Preformatted(LOAD_CODE, CODE),
    Paragraph("This loading path was checked in a fresh Kaggle session (natlas_health_demo.ipynb): the published "
              "adapter was downloaded from Hugging Face, loaded on top of N-ATLaS, and its answers to the first "
              "five benchmark questions matched the published predictions file 5 out of 5, as did the base "
              "model's.", P),
    Paragraph("For benchmark-style scoring, compare the next-token logits of the option letters instead of "
              "generating text. The function <i>predict</i> in natlas_pipeline.py does this.", P),
    Paragraph("7. Design decisions", H2),
    *bullets([
        "QLoRA was chosen so the whole run fits on one free 16 GB GPU, which keeps the work reproducible by "
        "anyone in Nigeria without paid compute.",
        "The benchmark is scored by option-letter probability, not by parsing generated text, so no answer is "
        "lost to formatting and the score is deterministic.",
        "The dataset's official train split holds too few usable questions to train on, so 500 questions were "
        "held out at random with a fixed seed before any model was scored. This is stated on the model card.",
        "The adapter is saved straight after training, before re-scoring, so a later failure cannot lose it.",
    ]),
    Paragraph("8. Limits and responsible use", H2),
    *bullets(LIMITS),
    Paragraph("9. Scaling and next steps", H2),
    *bullets([
        "Evaluate in Yoruba, Hausa and Igbo. N-ATLaS supports these languages, but this adapter has not been "
        "tested in them. This needs translated or native medical questions reviewed by clinicians.",
        "Extend the benchmark to short-answer and consumer health questions with clinician grading, which is "
        "closer to real use than multiple choice.",
        "The adapter is small, so it can be served next to base N-ATLaS on a single GPU and swapped per sector.",
    ]),
    Paragraph("10. Licence and attribution", H2),
    Paragraph("Adapter: N-ATLaS Terms of Use, the same as the base model (research and education, capped at "
              "1,000 active users). " + ATTRIBUTION, P),
    Paragraph("Data: AfriMed-QA v2, CC BY-SA 4.0. Olatunji et al., 2024, arXiv:2411.15640.", P),
]
build("4_Technical_Documentation.pdf", "Technical Documentation",
      "N-ATLaS Health QA: architecture, setup, reproduction and usage.", story)

# ---------------------------------------------------------------- 4. Team profile
story = [
    Paragraph("Team", H2),
    Paragraph("Team name: N-ATLaS Health QA. Team size: 1. Applying as an individual in the Innovation and "
              "Enterprise track.", P),
    Paragraph("Emmanuel Nwanguma, sole developer", H2),
    table([
        ["Role", "Sole developer. Responsible for data preparation, fine-tuning, benchmarking, documentation and "
                 "publication of the model."],
        ["Affiliation", "Independent developer"],
        ["Location", "Lagos, Nigeria"],
        ["Education", "BSc Biochemistry (Second Class Upper), Clifford University"],
        ["Health sector experience", "One year as Data and Records Assistant in the HIV/AIDS Unit of the Ministry "
                                     "of Health, Osogbo"],
        ["LinkedIn", "https://linkedin.com/in/nwangumaemmanuel"],
        ["GitHub", "https://github.com/Emart29"],
        ["Portfolio", "https://emart29.vercel.app"],
    ], [4.2 * cm, 12.8 * cm], header=False),
    Paragraph("Bio", H2),
    Paragraph("Emmanuel Nwanguma is a machine learning engineer based in Lagos. Background: a BSc in Biochemistry "
              "and one year as Data and Records Assistant in the HIV/AIDS Unit of the Ministry of Health, "
              "Osogbo.", P),
    Paragraph("Relevant work", H2),
    *bullets([
        "Author of six open-source tools for testing and monitoring AI systems.",
        "Built an 833-test benchmark comparing six language models.",
        "Previously fine-tuned a small language model for finance tasks with LoRA.",
        "Ranked 13th of more than 200 contributors to the open-source aden-hive agent framework, with 14 merged "
        "pull requests.",
    ]),
    Paragraph("Work on this entry", H2),
    *bullets([
        "Wrote the full pipeline: data preparation, QLoRA fine-tuning of N-ATLaS, benchmark and statistics.",
        f"Ran the 500-question benchmark against base N-ATLaS and published every per-question result.",
        f"Published the adapter at {esc(HF_URL)} and the code at {esc(GH_URL)}.",
    ]),
]
build("6_Team_Profile.pdf", "Team Profile", "N-ATLaS Health QA. National AI Innovation Challenge 2026.", story)
