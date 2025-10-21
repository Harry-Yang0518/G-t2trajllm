#!/usr/bin/env python3
import os
import json
import time
import argparse
import unicodedata
import re
from glob import glob

import pandas as pd
import jsonlines
from tqdm import tqdm
import json_repair
import random
from Levenshtein import distance,ratio
from scipy import stats
from openai import OpenAI

# ---------------------------
# Paths & CLI
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--texts", default="datasets/celebrity_biographies/texts.txt")
    p.add_argument("--labels", default="datasets/celebrity_biographies/labels.txt")
    p.add_argument("--preds", default="datasets/celebrity_biographies/preds.txt")
    p.add_argument("--run-inference", type=int, default=0, help="1 to re-run model inference")
    p.add_argument("--model", default="gpt-4-turbo")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=0.7)
    p.add_argument("--logdir", default="logs/celebrity_biographies")
    p.add_argument("--limit", type=int, default=0, help="limit number of examples (0 = all)")
    return p.parse_args()

# ---------------------------
# OpenAI client
# ---------------------------
client = OpenAI()  # reads OPENAI_API_KEY from env

def chatbot(prompt, text, model="gpt-4-turbo", top_p=0.7, temperature=0.1):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        top_p=top_p,
        temperature=temperature
    )
    return completion.choices[0].message.content

# ---------------------------
# Helpers (dedup / parse)
# ---------------------------
def remove_adjacent_duplicates(lst):
    if not lst:
        return []
    new_list = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            new_list.append(lst[i])
    return new_list

def json2places(track_json):
    """
    Expecting a list like:
    [
      {"Actor": "...", "Track": [{"Place": "..."} , ...]},
      ...
    ]
    Returns the 'best' actor's place sequence with adjacent duplicates removed.
    """
    try:
        dfs = []
        for actor in track_json:
            if isinstance(actor, str):
                continue
            tract = [track for track in actor.get("Track", []) if isinstance(track, dict)]
            df = pd.DataFrame(tract)
            df['Actor'] = actor.get('Actor', 'Unknown')
            dfs.append(df)
        if not dfs:
            return []
        dfs = pd.concat(dfs, ignore_index=True)
        if "Place" not in dfs.columns:
            dfs['Place'] = "XXX"
        dfs = dfs[(dfs['Place'] != 'XXX') & (dfs['Place'] != 'Unknown') & (dfs['Place'] != '')].reset_index(drop=True)
        if dfs.empty:
            return []

        # choose actor with the longest cleaned track
        best_track = []
        for actor_name, sub in dfs.groupby('Actor'):
            track = remove_adjacent_duplicates([str(p) for p in sub['Place'].tolist() if p])
            if len(track) > len(best_track):
                best_track = track
        return best_track
    except Exception:
        return []

def get_label(jsonl_file):
    jsonl = jsonlines.open(jsonl_file)
    df = pd.DataFrame(jsonl)
    df['label'] = df['label'].apply(lambda x: json2places(x))
    return df

def get_pred(jsonl_file):
    jsonl = jsonlines.open(jsonl_file)
    df = pd.DataFrame(jsonl)
    df['pred'] = df['pred'].apply(lambda x: json2places(x))
    return df

# ---------------------------
# Token-level edit distance
# ---------------------------
def token_levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        ai = a[i-1]
        for j in range(1, m+1):
            cost = 0 if ai == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[n][m]

# ---------------------------
# (Optional) CJK-safe normalizer
# Keep disabled by default for celebrity dataset unless you need it.
# ---------------------------
def normalize_place(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    # For multilingual bios you may want to preserve case; most names are proper nouns.
    # If you want case-insensitive matching, uncomment next line:
    # s = s.lower()
    # Remove punctuation except word chars, spaces, and CJK
    s = re.sub(r"[^\w\u4e00-\u9fff\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------
# Main metric with VERBOSE logging
# ---------------------------
def cal_index_verbose(index, label_raw, pred_raw, log_handle=None, print_everything=True):
    # Normalize
    label = [normalize_place(x) for x in label_raw if x]
    pred  = [normalize_place(x) for x in pred_raw  if x]

    LN, PN = len(label), len(pred)
    TP = 0

    sequences_l = label.copy()
    sequences_p = pred.copy()

    match_flags  = [False] * PN
    match_labels = [None] * LN
    match_preds  = [None] * PN

    match_events = []  # for logging

    # Greedy one-pass matching with exact | fuzzy | containment
    for i, l in enumerate(label):
        best = None
        for j, p in enumerate(pred):
            if match_flags[j]:
                continue
            sim = ratio(l, p)
            cond_contain = (l and p and (l in p or p in l))
            ok = (l == p) or (sim >= 0.80) or cond_contain
            if ok:
                best = (i, j, l, p, sim, cond_contain)
                break
        if best is not None:
            i2, j2, l2, p2, sim2, contain2 = best
            TP += 1
            match_flags[j2]  = True
            sequences_p[j2]  = l2
            match_labels[i2] = l2
            match_preds[j2]  = l2
            match_events.append({
                "label_idx": i2, "pred_idx": j2,
                "label_tok": l2, "pred_tok": p2,
                "lev_ratio": sim2, "containment": contain2,
                "reason": "exact" if l2 == p2 else ("containment" if contain2 else "ratio>=0.80")
            })

    # Kendall tau on positions
    idx_map = {}
    for j, tok in enumerate(match_preds):
        if tok is None:
            continue
        idx_map.setdefault(tok, []).append(j)

    pred_idx_order = []
    for tok in match_labels:
        if tok is None:
            continue
        if tok in idx_map and idx_map[tok]:
            pred_idx_order.append(idx_map[tok].pop(0))

    if len(pred_idx_order) >= 2:
        Kend = stats.kendalltau(range(len(pred_idx_order)), pred_idx_order)[0]
        Kendall_tau = 0.0 if str(Kend) == 'nan' else float(Kend)
    else:
        Kendall_tau = 0.0

    # Token edit distance
    dis = token_levenshtein(sequences_l, sequences_p)
    denom = (len(sequences_l) + len(sequences_p))
    edit_simlar = 1 - dis / denom if denom > 0 else 0.0

    # Per-item precision/recall/f1 (based on TP, LN, PN)
    prec_i = TP / PN if PN else 0.0
    rec_i  = TP / LN if LN else 0.0
    f1_i   = 0.0 if (prec_i + rec_i) == 0 else 2 * prec_i * rec_i / (prec_i + rec_i)

    # --------- Logging ---------
    log_record = {
        "id": index,
        "label_raw": label_raw,
        "pred_raw": pred_raw,
        "label_norm": label,
        "pred_norm": pred,
        "matches": match_events,
        "kendall_pred_idx_order": pred_idx_order,
        "Kendall_tau": Kendall_tau,
        "token_edit_distance": dis,
        "edit_similarity": edit_simlar,
        "TP": TP, "LN": LN, "PN": PN,
        "precision_item": prec_i, "recall_item": rec_i, "f1_item": f1_i,
        "unmatched_label": [tok for tok, m in zip(label, match_labels) if m is None],
        "unmatched_pred":  [p for p, f in zip(pred, match_flags) if not f],
    }

    if log_handle is not None:
        log_handle.write(log_record)

    if print_everything:
        print(f"\n=== ID: {index} ===")
        print(f"Label (raw): {label_raw}")
        print(f"Pred  (raw): {pred_raw}")
        print(f"Label (norm): {label}")
        print(f"Pred  (norm): {pred}")
        print(f"Matches ({TP}):")
        for m in match_events:
            print(f"  - L[{m['label_idx']}]='{m['label_tok']}'  <->  "
                  f"P[{m['pred_idx']}]='{m['pred_tok']}'  "
                  f"(ratio={m['lev_ratio']:.3f}, contain={m['containment']}, reason={m['reason']})")
        print(f"Kendall order: {pred_idx_order}  |  Kendall τ: {Kendall_tau:.4f}")
        print(f"Token edit distance: {dis}  |  edit_similarity: {edit_simlar:.4f}")
        print(f"Item precision={prec_i:.4f}, recall={rec_i:.4f}, f1={f1_i:.4f}")
        if log_record["unmatched_label"]:
            print(f"Unmatched label tokens: {log_record['unmatched_label']}")
        if log_record["unmatched_pred"]:
            print(f"Unmatched pred tokens:  {log_record['unmatched_pred']}")

    return (index, TP, LN, PN, Kendall_tau, dis, edit_simlar)

def cal_index_df(merge, log_path=None, print_everything=True):
    # JSONL logger
    log_writer = jsonlines.open(log_path, mode="w") if log_path else None
    rows = []
    for _, row in merge.iterrows():
        idx, TP, LN, PN, Kendall_tau, dis, edit_simlar = cal_index_verbose(
            row['id'], row['label'], row['pred'], log_handle=log_writer, print_everything=print_everything
        )
        rows.append((idx, TP, LN, PN, Kendall_tau, dis, edit_simlar))
    if log_writer:
        log_writer.close()

    index = pd.DataFrame(rows, columns=['id', 'TP', 'LN', 'PN', 'Kendall_tau', 'dis', 'edit_simlar']).set_index('id')
    return index

def cal_index_all(merge, log_path=None, print_everything=True):
    index = cal_index_df(merge, log_path=log_path, print_everything=print_everything)
    precision = index.TP.sum() / index.PN.sum() if index.PN.sum() else 0.0
    recall    = index.TP.sum() / index.LN.sum() if index.LN.sum() else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    edit_simlar = float(index.edit_simlar.mean()) if len(index) else 0.0
    kendall_tau = float((index.Kendall_tau * index.TP).sum() / index.TP.sum()) if index.TP.sum() else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1, 'kendall_tau': kendall_tau, 'edit_simlar': edit_simlar}, index

# ---------------------------
# Inference prompt template (reuse yours)
# ---------------------------
def build_prompt_template(args):
    # 1. Sampling-induction process

    ## Diverse knowledge for trajectory extraction
    ### 3.2.1.Text-to-trajectory transformation model
    tran_model = '''
    In narratology, the narrative space of a text is typically divided into three vertical levels: textual space, topographical space, and chronotopic space. Textual space refers to the linear form of narrative expression and serves as the direct medium for conveying the narrative. Topographical space represents the static spatial entities in the narrative, such as cities or villages, which are presented in the text as location descriptions. Chronotopic space encompasses the dynamic spatiotemporal elements, including events and movements. 
    Based on narrative spatial theory, extracting trajectories from narrative texts involves transforming trajectory information from the textual space to the chronotopic space. However, the textual space compresses the information from topographical and chronotopic spaces due to its linear structure and selective expression, making this transformation challenging to achieve directly. Inspired by existing trajectory extraction methods that break down the process, we decompose the transformation into the following sub-transformations.
    •Element transformation: the linear textual representations in the textual space are converted into descriptions of static geographical entities within the topographical space.
    •Actor-centric transformation: actors in the textual space are linked with the static geographical entities in the topographical space.
    •Ordering transformation: the sequence of movements by the actor between different geographical entities is captured within the chronotopic space.
    '''
    ### 3.2.2.Simplified trajectory model
    data_model = '''```json
[
    {
        "Actor": XXX,
        "Track": [{'Place': XXX}, ...]
    },
...
]```'''

    ### 3.2.3.Manually annotated text-trajectory pairs
    def random_sample(args):
        text_df = pd.DataFrame(jsonlines.open(args.texts))
        label_df = pd.DataFrame(jsonlines.open(args.labels))
        data = pd.merge(text_df, label_df,on='id')
        sample = data.sample(1)
        return sample.text, sample.label


    ### 3.3 sample from LLM

    res = []
    for i in range(3):
        text, traj = random_sample(args)
        prompt = '''Propose a step-by-step Reasoning Path and Constraints for extracting trajectory from text based on the Transformation Model and Text-trajectory Example.'''
        input_for_sample = f'''
## Transformation Model: {tran_model}
## Text-trajectory Example: 
## Text: {text}
## Trajectory: {traj}'''
        res.append(chatbot(prompt,input_for_sample))

    ### 3.3 Combine induction’s Reasoning Path and Constraints with prompt template
    #### this is a sample of the induction的Reasoning Path and Constraints

    rea_path = '''
1. Identify all entities of Places, and Times.
2. Determine whether each Place is a location actually visited.
3. Construct the trajectory of the Actor according to the order of visits.
'''
    constraints = '''
- Ensure that each trajectory point is a single location; if not, split it into multiple trajectory points.
- Ensure that the trajectory points are visited by the Actor; if not, remove them from the trajectory.
- Ensure that the trajectory points are arranged in the order of visits; if not, rearrange them.
'''

    prompt_template = f'''
# Task
- Extract trajectory from the following text.

## Steps
{rea_path}

## Reflection
{constraints}

## Output
{data_model}
'''

    return prompt_template

# ---------------------------
# Main
# ---------------------------
def main():
    args = get_args()
    os.makedirs(args.logdir, exist_ok=True)
    raw_dir = os.path.join(args.logdir, "preds_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 1) Run inference if requested
    if args.run_inference:
        text_df = pd.DataFrame(jsonlines.open(args.texts))
        if args.limit > 0:
            text_df = text_df.head(args.limit)

        pred_df = pd.DataFrame(columns=['id', 'pred'])
        prompt_template = build_prompt_template(args)

        for _id, text in tqdm(text_df[['id', 'text']].values, desc="Running inference"):
            try:
                output = chatbot(prompt_template, text, model=args.model, top_p=args.top_p, temperature=args.temperature)
                # Try robust JSON repair
                parsed = json_repair.loads(output)

                # Normalize to list[dict]
                if isinstance(parsed, dict):
                    actors = [parsed]
                elif isinstance(parsed, list):
                    actors = [o for o in parsed if isinstance(o, dict)]
                else:
                    actors = []

                # Save raw LLM output for debugging
                with open(os.path.join(raw_dir, f"{_id}.json"), "w", encoding="utf-8") as f:
                    json.dump({"id": _id, "raw_output": output, "parsed": actors}, f, ensure_ascii=False, indent=2)

                pred_df = pd.concat([pred_df, pd.DataFrame([{'id': _id, 'pred': actors}])], ignore_index=True)

            except Exception as e:
                print(f"[ERROR] id={_id}: {e}")

            time.sleep(0.3)  # be gentle

        pred_df.to_json(args.preds, orient='records', lines=True, force_ascii=False)

    else:
        if not os.path.exists(args.preds):
            raise FileNotFoundError(f"Preds file not found at {args.preds}. Use --run-inference=1 to generate it.")

    # 2) Load labels & preds; merge; evaluate with detailed logging
    label_df = get_label(args.labels)
    pred_df  = get_pred(args.preds)

    if args.limit > 0:
        # limit AFTER loading/merging to keep id alignment
        tmp = pd.merge(label_df, pred_df, on='id', how='inner')
        tmp = tmp.head(args.limit)
    else:
        tmp = pd.merge(label_df, pred_df, on='id', how='inner')

    debug_jsonl = os.path.join(args.logdir, "metrics_debug.jsonl")
    summary, index_df = cal_index_all(tmp, log_path=debug_jsonl, print_everything=True)

    print("\nOverall:")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    # Save the per-id table as CSV for browsing
    per_id_csv = os.path.join(args.logdir, "per_id_metrics.csv")
    index_df.to_csv(per_id_csv, encoding="utf-8")
    print(f"\nSaved per-id metrics to: {per_id_csv}")
    print(f"Saved detailed JSONL logs to: {debug_jsonl}")
    print(f"Saved raw model outputs to: {raw_dir}")

if __name__ == "__main__":
    main()
