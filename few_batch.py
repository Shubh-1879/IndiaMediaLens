# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Few-shot stance classification (entity/statement) — FINAL JSON OUTPUT (Option B, braces fixed) — BATCHED + RESUME

# - Few-shot only.
# - Batched generation via raw HF pipeline(list_of_prompts, batch_size=...).
# - Resume: --resume continues from existing --output_csv, skipping rows with a filled fewshot_label.
# - Model returns ONLY JSON: {"stance":"<label>","reason":"<short phrase>"}.
# - Prevents prompt-echo (return_full_text=False); deterministic decoding.
# - Robust JSON extraction + hardened fallback normalizer.
# - Adds mapped column: supports→for, denies→against.

# Usage:
# python fewshot_stance_json.py \
#   --input_csv /path/to/your.csv \
#   --model /path/to/local/hf/snapshot \
#   --shots_json /path/to/few_shot_examples.json \
#   --output_csv ./results/out.csv \
#   --max_new_tokens 48 \
#   --batch_size 16 \
#   --bucket_by_length \
#   --resume \
#   --save_every 100
# """

# import os, json, argparse, time, logging, warnings, re
# import pandas as pd
# import numpy as np
# from tqdm import tqdm

# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# # LangChain imports (new style first, fallback to older) — used only for prompt construction
# try:
#     from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
# except ImportError:
#     from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# warnings.filterwarnings(
#     "ignore",
#     message="You seem to be using the pipelines sequentially on GPU."
# )

# # ---------------------------
# # HF model wrapper (RAW PIPELINE for batching)
# # ---------------------------

# def build_hf_pipe(model_path: str, task_hint: str = "auto", max_new_tokens: int = 48):
#     """
#     Returns a raw HF pipeline that supports batched calls: pipe(list_of_prompts, batch_size=..., padding=True, truncation=True)
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#     # Set pad_token_id if missing
#     if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     # Try to auto-detect model class to pick task
#     task = None
#     if task_hint in ("text-generation", "text2text-generation"):
#         task = task_hint
#     else:
#         # Heuristic: if an encoder-decoder config exists, choose text2text
#         try:
#             _ = AutoModelForSeq2SeqLM.from_pretrained(
#                 model_path,
#                 torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
#                 device_map="auto"
#             )
#             task = "text2text-generation"
#         except Exception:
#             task = "text-generation"

#     # (Optional) faster matmul on Ampere+
#     try:
#         torch.set_float32_matmul_precision("high")
#     except Exception:
#         pass

#     kwargs = dict(
#         model=model_path,
#         tokenizer=tokenizer,
#         torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
#         device_map="auto",
#         max_new_tokens=max_new_tokens,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         # Deterministic
#         do_sample=False,
#         temperature=0.0,
#         top_p=1.0,
#         repetition_penalty=1.0,
#     )
#     if task == "text-generation":
#         kwargs["return_full_text"] = False

#     return pipeline(task, **kwargs)

# # ---------------------------
# # Few-shot examples + label utils
# # ---------------------------

# def load_shots(shots_json_path: str):
#     with open(shots_json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     for i, ex in enumerate(data):
#         for k in ("entity", "statement", "stance"):
#             if k not in ex:
#                 raise ValueError(f"Example {i} missing key '{k}'. Found keys: {list(ex.keys())}")
#         if ex["stance"] not in {"supports", "denies", "neutral", "unrelated"}:
#             raise ValueError(
#                 f"Example {i} has invalid 'stance'={ex['stance']}. "
#                 f"Must be one of supports|denies|neutral|unrelated."
#             )
#     return data

# def normalize_label(text: str) -> str:
#     t = (text or "").strip().lower()
#     first = re.split(r'[\s\|\.,:;()\[\]\{\}\n\r\t"]+', t)[0]
#     if first in {"supports", "denies", "neutral", "unrelated"}:
#         return first
#     m = re.search(r'\b(supports|denies|neutral|unrelated)\b', t)
#     if m:
#         return m.group(1)
#     if re.search(r'\b(deny|denies|against|anti)\b', t):
#         return "denies"
#     if re.search(r'\b(supports?|pro)\b', t):
#         return "supports"
#     if re.search(r'\bneutral\b', t):
#         return "neutral"
#     if re.search(r'\b(unrelated|irrelevant|off-?topic)\b', t):
#         return "unrelated"
#     return "neutral"

# def extract_json(s: str):
#     if not s:
#         return None, None
#     try:
#         m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
#         if not m:
#             return None, None
#         obj = json.loads(m.group(0))
#         stance = obj.get("stance")
#         reason = obj.get("reason", "")
#         if stance in {"supports", "denies", "neutral", "unrelated"}:
#             return stance, str(reason)
#     except Exception:
#         pass
#     return None, None

# # ---------------------------
# # Prompt construction
# # ---------------------------

# def make_few_shot_prompt(examples):
#     example_template = (
#         "entity: {entity}\n"
#         "statement: {statement}\n"
#         "stance: {stance}"
#     )
#     example_prompt = PromptTemplate(
#         input_variables=["entity", "statement", "stance"],
#         template=example_template,
#     )

#     prefix = (
#         "Stance classification is the task of determining the expressed or implied opinion, "
#         "or stance, of a statement toward a certain, specified target. The following "
#         "statements are social media posts expressing opinions about entities. "
#         "Each statement can either support, deny, be neutral, or be unrelated toward its entity."
#     )

#     # NOTE: braces in the example JSON are escaped as {{ and }} to keep them literal.
#     suffix = (
#         "Analyze the following social media statement and determine its stance towards the provided entity.\n"
#         "Return ONLY a compact JSON object with exactly these keys:\n"
#         '- \"stance\": one of \"supports\", \"denies\", \"neutral\", \"unrelated\"\n'
#         '- \"reason\": a short phrase (not a paragraph)\n'
#         'Example: {{\"stance\":\"supports\",\"reason\":\"praises aid to the group\"}}\n'
#         "entity: {event}\n"
#         "statement: {statement}\n"
#         "JSON:"
#     )

#     return FewShotPromptTemplate(
#         examples=examples,
#         example_prompt=example_prompt,
#         prefix=prefix,
#         suffix=suffix,
#         input_variables=["event", "statement"],
#         example_separator="\n\n",
#     )

# # ---------------------------
# # Main
# # ---------------------------

# def main():
#     ap = argparse.ArgumentParser(description="Few-shot stance classifier (entity/statement) — JSON output (batched + resume).")
#     ap.add_argument("--input_csv", required=True, help="Input CSV with columns: tweet,...,keyword")
#     ap.add_argument("--model", required=True, help="Local HF snapshot path (decoder or encoder-decoder).")
#     ap.add_argument("--shots_json", required=True, help="JSON with few-shot examples.")
#     ap.add_argument("--output_csv", required=True, help="Output CSV path.")
#     ap.add_argument("--task_hint", default="auto", choices=["auto", "text-generation", "text2text-generation"],
#                     help="Force HF pipeline task if needed.")
#     ap.add_argument("--max_new_tokens", type=int, default=48)
#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--bucket_by_length", action="store_true", help="Sort prompts by length within the run for less padding.")
#     ap.add_argument("--resume", action="store_true", help="Resume from existing --output_csv if present; skip already-scored rows.")
#     ap.add_argument("--save_every", type=int, default=100, help="Write partial CSV every N processed rows.")
#     ap.add_argument("--log_file", default=None)
#     args = ap.parse_args()

#     # Logging
#     logger = logging.getLogger("fewshot_stance_json")
#     logger.setLevel(logging.INFO)
#     logger.handlers.clear()
#     sh = logging.StreamHandler()
#     sh.setLevel(logging.INFO)
#     logger.addHandler(sh)
#     if args.log_file:
#         fh = logging.FileHandler(args.log_file)
#         fh.setLevel(logging.INFO)
#         logger.addHandler(fh)

#     # Load data
#     df = pd.read_csv(args.input_csv)
#     if not {"tweet", "keyword"}.issubset(df.columns):
#         raise ValueError("Input CSV must contain at least 'tweet' and 'keyword' columns.")

#     # Output columns
#     raw_col    = "fewshot_raw"
#     norm_col   = "fewshot_label"               # supports | denies | neutral | unrelated
#     mapped_col = "fewshot_label_for_against"   # for | against | neutral | unrelated
#     reason_col = "fewshot_reason"              # short explanation (from JSON)

#     # Prepare outputs (with resume support)
#     if args.resume and os.path.exists(args.output_csv):
#         try:
#             df_out = pd.read_csv(args.output_csv)
#             if len(df_out) != len(df):
#                 logger.warning("Resume requested but output length != input length. Starting fresh.")
#                 df_out = df.copy()
#             else:
#                 # Ensure columns exist
#                 if "stance_gold" not in df_out.columns:
#                     df_out["stance_gold"] = np.nan
#                 for c in (raw_col, norm_col, mapped_col, reason_col):
#                     if c not in df_out.columns:
#                         df_out[c] = ""
#                 logger.info(f"Resuming from existing file: {args.output_csv}")
#         except Exception as e:
#             logger.warning(f"Failed to read existing output for resume ({e}). Starting fresh.")
#             df_out = df.copy()
#     else:
#         df_out = df.copy()

#     # If fresh, initialize columns
#     if "stance_gold" not in df_out.columns:
#         df_out["stance_gold"] = np.nan
#     for c in (raw_col, norm_col, mapped_col, reason_col):
#         if c not in df_out.columns:
#             df_out[c] = ""

#     # Build pipeline (batched)
#     pipe = build_hf_pipe(args.model, task_hint=args.task_hint, max_new_tokens=args.max_new_tokens)

#     # Load few-shot examples & prompt
#     shots = load_shots(args.shots_json)
#     prompt = make_few_shot_prompt(shots)

#     # Determine which rows still need scoring
#     valid_labels = {"supports", "denies", "neutral", "unrelated"}
#     if norm_col in df_out.columns:
#         done_mask = df_out[norm_col].astype(str).str.strip().str.lower().isin(valid_labels)
#     else:
#         done_mask = pd.Series(False, index=df_out.index)

#     remaining_idx = df_out.index[~done_mask].tolist()
#     if len(remaining_idx) == 0:
#         logger.info("All rows already scored. Nothing to do.")
#         # still ensure file exists and is up to date
#         df_out.to_csv(args.output_csv, index=False)
#         return

#     # Build prompts only for remaining rows
#     prompts = []
#     row_indices = []
#     for i in remaining_idx:
#         row = df_out.loc[i]
#         entity = str(row["keyword"]).strip()
#         statement = str(row["tweet"]).replace("\n", " ").strip()
#         prompts.append(prompt.format(event=entity, statement=statement))
#         row_indices.append(i)

#     # Optional: bucket by length to reduce padding
#     if args.bucket_by_length:
#         order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
#         prompts = [prompts[k] for k in order]
#         row_indices = [row_indices[k] for k in order]

#     # Batched inference
#     start = time.time()
#     processed = 0
#     last_saved = 0
#     B = max(1, int(args.batch_size))

#     for start_idx in tqdm(range(0, len(prompts), B), desc="Scoring (batched)"):
#         batch_prompts = prompts[start_idx:start_idx + B]
#         batch_rows    = row_indices[start_idx:start_idx + B]

#         outs = pipe(
#             batch_prompts,
#             batch_size=B,
#             padding=True,
#             truncation=True
#         )

#         # Normalize outputs to strings
#         texts = []
#         for out in outs:
#             # Depending on pipeline version, out can be a list[{"generated_text": ...}] or dict
#             if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
#                 text = out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
#             elif isinstance(out, dict):
#                 text = out.get("generated_text") or out.get("summary_text") or str(out)
#             else:
#                 text = str(out)
#             texts.append(text)

#         # Parse + write back
#         label_map = {
#             "supports": "for",
#             "denies": "against",
#             "neutral": "neutral",
#             "unrelated": "unrelated",
#         }

#         for i_row, text in zip(batch_rows, texts):
#             stance_json, reason = extract_json(text)
#             if stance_json:
#                 norm = stance_json
#                 df_out.at[i_row, reason_col] = (reason or "").strip()
#             else:
#                 norm = normalize_label(text)

#             df_out.at[i_row, raw_col]    = text
#             df_out.at[i_row, norm_col]   = norm
#             df_out.at[i_row, mapped_col] = label_map.get(norm, "neutral")

#         processed += len(batch_rows)
#         if processed - last_saved >= args.save_every:
#             df_out.to_csv(args.output_csv, index=False)
#             last_saved = processed

#     # Final save
#     df_out.to_csv(args.output_csv, index=False)
#     logger.info(f"Done in {time.time() - start:.1f}s. Scored {processed} rows this run. Wrote: {args.output_csv}")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Few-shot stance classification (entity/statement) — FINAL JSON OUTPUT (Option B, braces fixed) — BATCHED + RESUME

- Few-shot only.
- Batched generation via raw HF pipeline(list_of_prompts, batch_size=...).
- Resume: --resume continues from existing --output_csv, skipping rows with a filled fewshot_label.
- Model returns ONLY JSON: {"stance":"<label>","reason":"<short phrase>"}.
- Prevents prompt-echo (return_full_text=False); deterministic decoding.
- Robust JSON extraction + hardened fallback normalizer.
- Adds mapped column: supports→for, denies→against.
- UPDATED: per-keyword few-shot selection. Loads shots from --shots_dir/<prefix>_<keyword>_stance.json
  (keyword is slugified). Falls back to --shots_json if the per-keyword file is missing.
"""

import os, json, argparse, time, logging, warnings, re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# LangChain imports (new style first, fallback to older) — used only for prompt construction
try:
    from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate

warnings.filterwarnings(
    "ignore",
    message="You seem to be using the pipelines sequentially on GPU."
)

# ---------------------------
# HF model wrapper (RAW PIPELINE for batching)
# ---------------------------

def build_hf_pipe(model_path: str, task_hint: str = "auto", max_new_tokens: int = 48):
    """
    Returns a raw HF pipeline that supports batched calls: pipe(list_of_prompts, batch_size=..., padding=True, truncation=True)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Set pad_token_id if missing
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Try to auto-detect model class to pick task
    task = None
    if task_hint in ("text-generation", "text2text-generation"):
        task = task_hint
    else:
        # Heuristic: if an encoder-decoder config exists, choose text2text
        try:
            _ = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                device_map="auto"
            )
            task = "text2text-generation"
        except Exception:
            task = "text-generation"

    # (Optional) faster matmul on Ampere+
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    kwargs = dict(
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Deterministic
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    if task == "text-generation":
        kwargs["return_full_text"] = False

    return pipeline(task, **kwargs)

# ---------------------------
# Few-shot examples + label utils
# ---------------------------

def _canon_label(s: str) -> str:
    return re.sub(r'[^a-z]+', '', (s or '').strip().lower())

def _slugify_kw(s: str) -> str:
    slug = re.sub(r'[^0-9a-zA-Z]+', '_', (s or '').strip().lower()).strip('_')
    return slug or "unknown"

# Accepts your older few-shot JSONs that may use positive/negative/neutral
_LABEL_MAP_IN = {
    "positive": "supports",
    "negative": "denies",
    "for": "supports",
    "against": "denies",
}
_VALID = {"supports", "denies", "neutral", "unrelated"}

def _normalize_shot_label(s: str) -> str:
    c = _canon_label(s)
    if c in _VALID:
        return c
    if c in _LABEL_MAP_IN:
        return _LABEL_MAP_IN[c]
    # last resort: neutral
    return "neutral"

def load_shots_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for i, ex in enumerate(data):
        # tolerate minor key variations
        entity = ex.get("entity")
        statement = ex.get("statement")
        stance = ex.get("stance")
        if entity is None or statement is None or stance is None:
            raise ValueError(f"Few-shot example #{i} in {path} missing required keys.")
        stance = _normalize_shot_label(stance)
        if stance not in _VALID:
            stance = "neutral"
        out.append({"entity": entity, "statement": statement, "stance": stance})
    return out

def normalize_label(text: str) -> str:
    t = (text or "").strip().lower()
    first = re.split(r'[\s\|\.,:;()\[\]\{\}\n\r\t"]+', t)[0]
    if first in {"supports", "denies", "neutral", "unrelated"}:
        return first
    m = re.search(r'\b(supports|denies|neutral|unrelated)\b', t)
    if m:
        return m.group(1)
    if re.search(r'\b(deny|denies|against|anti)\b', t):
        return "denies"
    if re.search(r'\b(supports?|pro)\b', t):
        return "supports"
    if re.search(r'\bneutral\b', t):
        return "neutral"
    if re.search(r'\b(unrelated|irrelevant|off-?topic)\b', t):
        return "unrelated"
    return "neutral"

def extract_json(s: str):
    if not s:
        return None, None
    try:
        # strip ```json ... ``` if present
        mcode = re.search(r"```json(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
        if mcode:
            s = mcode.group(1)
        m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
        if not m:
            return None, None
        obj = json.loads(m.group(0))
        stance = obj.get("stance")
        reason = obj.get("reason", "")
        if stance in {"supports", "denies", "neutral", "unrelated"}:
            return stance, str(reason)
    except Exception:
        pass
    return None, None

# ---------------------------
# Prompt construction
# ---------------------------

def make_few_shot_prompt(examples):
    example_template = (
        "entity: {entity}\n"
        "statement: {statement}\n"
        "stance: {stance}"
    )
    example_prompt = PromptTemplate(
        input_variables=["entity", "statement", "stance"],
        template=example_template,
    )

    prefix = (
        "Stance classification is the task of determining the expressed or implied opinion, "
        "or stance, of a statement toward a certain, specified target. The following "
        "statements are social media posts expressing opinions about entities. "
        "Each statement can either support, deny, be neutral, or be unrelated toward its entity."
    )

    # NOTE: braces in the example JSON are escaped as {{ and }} to keep them literal.
    suffix = (
        "Analyze the following social media statement and determine its stance towards the provided entity.\n"
        "Return ONLY a compact JSON object with exactly these keys:\n"
        '- \"stance\": one of \"supports\", \"denies\", \"neutral\", \"unrelated\"\n'
        '- \"reason\": a short phrase (not a paragraph)\n'
        'Example: {{\"stance\":\"supports\",\"reason\":\"praises aid to the group\"}}\n'
        "entity: {event}\n"
        "statement: {statement}\n"
        "JSON:"
    )

    return FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["event", "statement"],
        example_separator="\n\n",
    )

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Few-shot stance classifier (entity/statement) — JSON output (batched + resume).")
    ap.add_argument("--input_csv", required=True, help="Input CSV with columns: tweet,...,keyword")
    ap.add_argument("--model", required=True, help="Local HF snapshot path (decoder or encoder-decoder).")
    # Per-keyword few-shot config
    ap.add_argument("--shots_dir", required=True, help="Directory containing per-keyword few-shot JSONs like <prefix>_<keyword>_stance.json")
    ap.add_argument("--shots_prefix", default="kyra", help="Prefix for few-shot JSON filenames (default: kyra)")
    # Fallback (optional) single shots file if per-keyword file is missing
    ap.add_argument("--shots_json", default=None, help="Fallback JSON with few-shot examples when per-keyword file is missing.")
    ap.add_argument("--output_csv", required=True, help="Output CSV path.")
    ap.add_argument("--task_hint", default="auto", choices=["auto", "text-generation", "text2text-generation"],
                    help="Force HF pipeline task if needed.")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--bucket_by_length", action="store_true", help="Sort prompts by length within the run for less padding.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing --output_csv if present; skip already-scored rows.")
    ap.add_argument("--save_every", type=int, default=100, help="Write partial CSV every N processed rows.")
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    # Logging
    logger = logging.getLogger("fewshot_stance_json")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

    shots_dir = Path(args.shots_dir).expanduser().resolve()
    if not shots_dir.exists() or not shots_dir.is_dir():
        raise FileNotFoundError(f"--shots_dir not found or not a directory: {shots_dir}")

    # Load data
    df = pd.read_csv(args.input_csv)
    if not {"tweet", "keyword"}.issubset(df.columns):
        raise ValueError("Input CSV must contain at least 'tweet' and 'keyword' columns.")

    # Output columns
    raw_col    = "fewshot_raw"
    norm_col   = "fewshot_label"               # supports | denies | neutral | unrelated
    mapped_col = "fewshot_label_for_against"   # for | against | neutral | unrelated
    reason_col = "fewshot_reason"              # short explanation (from JSON)

    # Prepare outputs (with resume support)
    if args.resume and os.path.exists(args.output_csv):
        try:
            df_out = pd.read_csv(args.output_csv)
            if len(df_out) != len(df):
                logger.warning("Resume requested but output length != input length. Starting fresh.")
                df_out = df.copy()
            else:
                # Ensure columns exist
                if "stance_gold" not in df_out.columns:
                    df_out["stance_gold"] = np.nan
                for c in (raw_col, norm_col, mapped_col, reason_col):
                    if c not in df_out.columns:
                        df_out[c] = ""
                logger.info(f"Resuming from existing file: {args.output_csv}")
        except Exception as e:
            logger.warning(f"Failed to read existing output for resume ({e}). Starting fresh.")
            df_out = df.copy()
    else:
        df_out = df.copy()

    # If fresh, initialize columns
    if "stance_gold" not in df_out.columns:
        df_out["stance_gold"] = np.nan
    for c in (raw_col, norm_col, mapped_col, reason_col):
        if c not in df_out.columns:
            df_out[c] = ""

    # Build pipeline (batched)
    pipe = build_hf_pipe(args.model, task_hint=args.task_hint, max_new_tokens=args.max_new_tokens)

    # ---- Per-keyword shots/template cache ----
    shots_cache: dict[str, list] = {}
    tmpl_cache: dict[str, FewShotPromptTemplate] = {}

    # Optional global fallback shots
    fallback_shots = None
    if args.shots_json:
        fb_path = Path(args.shots_json).expanduser().resolve()
        if fb_path.exists():
            try:
                fallback_shots = load_shots_file(fb_path)
                logger.info(f"Loaded fallback shots: {fb_path} (n={len(fallback_shots)})")
            except Exception as e:
                logger.warning(f"Failed to load fallback shots {fb_path}: {e}")
        else:
            logger.warning(f"Fallback --shots_json not found: {fb_path}")

    def get_prompt_for_keyword(keyword_value: str) -> FewShotPromptTemplate:
        slug = _slugify_kw(keyword_value)
        if slug in tmpl_cache:
            return tmpl_cache[slug]
        # Resolve per-keyword file
        shots_path = shots_dir / f"{args.shots_prefix}_{slug}_stance.json"
        if shots_path.exists():
            shots = load_shots_file(shots_path)
            shots_cache[slug] = shots
            tmpl_cache[slug] = make_few_shot_prompt(shots)
            return tmpl_cache[slug]
        # Fallback to global shots if available
        if fallback_shots is not None:
            tmpl_cache[slug] = make_few_shot_prompt(fallback_shots)
            return tmpl_cache[slug]
        raise FileNotFoundError(
            f"No few-shot file for keyword '{keyword_value}' (slug '{slug}') at {shots_path}, "
            f"and no --shots_json fallback provided."
        )

    # Determine which rows still need scoring
    valid_labels = {"supports", "denies", "neutral", "unrelated"}
    if norm_col in df_out.columns:
        done_mask = df_out[norm_col].astype(str).str.strip().str.lower().isin(valid_labels)
    else:
        done_mask = pd.Series(False, index=df_out.index)

    remaining_idx = df_out.index[~done_mask].tolist()
    if len(remaining_idx) == 0:
        logger.info("All rows already scored. Nothing to do.")
        df_out.to_csv(args.output_csv, index=False)
        return

    # Build prompts only for remaining rows (per-keyword)
    prompts = []
    row_indices = []
    missing_shots_keywords = set()

    for i in remaining_idx:
        row = df_out.loc[i]
        entity = str(row["keyword"]).strip()
        statement = str(row["tweet"]).replace("\n", " ").strip()
        try:
            prompt_tmpl = get_prompt_for_keyword(entity)
        except FileNotFoundError as e:
            missing_shots_keywords.add(entity)
            raise
        prompts.append(prompt_tmpl.format(event=entity, statement=statement))
        row_indices.append(i)

    if missing_shots_keywords:
        logger.warning(f"Missing shots for keywords: {sorted(list(missing_shots_keywords))}")

    # Optional: bucket by length to reduce padding
    if args.bucket_by_length:
        order = sorted(range(len(prompts)), key=lambda k: len(prompts[k]))
        prompts = [prompts[k] for k in order]
        row_indices = [row_indices[k] for k in order]

    # Batched inference
    start = time.time()
    processed = 0
    last_saved = 0
    B = max(1, int(args.batch_size))

    for start_idx in tqdm(range(0, len(prompts), B), desc="Scoring (batched)"):
        batch_prompts = prompts[start_idx:start_idx + B]
        batch_rows    = row_indices[start_idx:start_idx + B]

        outs = pipe(
            batch_prompts,
            batch_size=B,
            padding=True,
            truncation=True
        )

        # Normalize outputs to strings
        texts = []
        for out in outs:
            # Depending on pipeline version, out can be a list[{"generated_text": ...}] or dict
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                text = out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
            elif isinstance(out, dict):
                text = out.get("generated_text") or out.get("summary_text") or str(out)
            else:
                text = str(out)
            texts.append(text)

        # Parse + write back
        label_map = {
            "supports": "for",
            "denies": "against",
            "neutral": "neutral",
            "unrelated": "unrelated",
        }

        for i_row, text in zip(batch_rows, texts):
            stance_json, reason = extract_json(text)
            if stance_json:
                norm = stance_json
                df_out.at[i_row, reason_col] = (reason or "").strip()
            else:
                norm = normalize_label(text)

            df_out.at[i_row, raw_col]    = text
            df_out.at[i_row, norm_col]   = norm
            df_out.at[i_row, mapped_col] = label_map.get(norm, "neutral")

        processed += len(batch_rows)
        if processed - last_saved >= args.save_every:
            df_out.to_csv(args.output_csv, index=False)
            last_saved = processed

    # Final save
    df_out.to_csv(args.output_csv, index=False)
    elapsed = time.time() - start
    logger.info(f"Done in {elapsed:.1f}s. Scored {processed} rows this run. Wrote: {args.output_csv}")

if __name__ == "__main__":
    main()
