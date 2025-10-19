from openai import OpenAI
from glob import glob
import pandas as pd
import time
import json
import os
from tqdm import tqdm
import json_repair
import random
from Levenshtein import distance,ratio
from scipy import stats
import jsonlines    
import re
import unicodedata
from Levenshtein import distance as char_distance, ratio as char_ratio
import numpy as np
from itertools import product
from Levenshtein import ratio as char_ratio
try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except Exception:
    HAS_HUNGARIAN = False


RUN_INFERENCE = False  # set True only when you want to re-run LLM inference
LABELS_PATH = "datasets/travel_blogs/labels.txt"
PREDS_PATH  = "datasets/travel_blogs/preds.txt"
# --- add near the imports ---



def place_sim(a: str, b: str) -> float:
    """Similarity between two normalized place names."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = char_ratio(a, b)
    # containment bonus (e.g., "Hague" in "The Hague")
    if a in b or b in a:
        r = max(r, 0.92)
    return r


def best_matches(labels, preds, thr=0.80):
    """
    Find a maximum-score 1-1 alignment between labels and preds
    and keep only edges with similarity >= thr.
    Returns: list of (i_label, j_pred, sim), sorted by j_pred (pred order).
    """
    if not labels or not preds:
        return []

    L, P = len(labels), len(preds)
    S = np.zeros((L, P), dtype=float)
    for i, j in product(range(L), range(P)):
        S[i, j] = place_sim(labels[i], preds[j])

    if HAS_HUNGARIAN:
        cost = 1.0 - S  # maximize S == minimize cost
        li, pj = linear_sum_assignment(cost)
        pairs = [(i, j, S[i, j]) for i, j in zip(li, pj) if S[i, j] >= thr]
    else:
        # fallback: greedy non-conflicting edges
        edges = sorted(((S[i, j], i, j) for i in range(L) for j in range(P)), reverse=True)
        usedL, usedP, pairs = set(), set(), []
        for s, i, j in edges:
            if s < thr: break
            if i in usedL or j in usedP: continue
            usedL.add(i); usedP.add(j); pairs.append((i, j, s))

    # stable order by prediction index (useful for downstream)
    pairs.sort(key=lambda x: x[1])
    return pairs


client = OpenAI()

def chatbot(prompt, text):
    completion = client.chat.completions.create(
        model="gpt-4-turbo",  # you can also use "gpt-4-turbo" if preferred
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        top_p=0.7,
        temperature=0.1
    )
    return completion.choices[0].message.content

# # Parse output and post-processing

def remove_adjacent_duplicates(lst):
    new_list = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            new_list.append(lst[i])
    return new_list

# Normalize a single place name conservatively
def normalize_place(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip()
    # unify to NFC (safe for CJK)
    s = unicodedata.normalize("NFC", s)
    # standardize parentheses variants to plain "()", then optionally drop inner note
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # collapse internal whitespace
    s = re.sub(r"\s+", "", s)
    return s

def json2places(track_json):
    try:
        dfs = []
        for actor in track_json:
            if type(actor) == str:
                continue
            tract = [track for track in actor.get("Track",[]) if type(track) == dict]
            df = pd.DataFrame(tract)
            df['Actor'] = actor.get('Actor', 'Actor')
            dfs.append(df)
        dfs = pd.concat(dfs) if dfs else pd.DataFrame(columns=["Place","Actor"])
        if "Place" not in dfs.columns:
            dfs['Place'] = "XXX"
        dfs = dfs[(dfs['Place'] != 'XXX') & (dfs['Place'] != 'Unknown') & (dfs['Place'] != '')].reset_index(drop=True)
        if dfs.empty:
            return []
        actor_counts = dfs['Actor'].value_counts()
        most_common_actor = actor_counts.idxmax()
        pred = dfs[dfs['Actor'] == most_common_actor].reset_index(drop=True).Place.to_list()

        # NEW: normalize and de-dup adjacent after normalization
        pred = [normalize_place(p) for p in pred if isinstance(p, str)]
        pred = [p for p in pred if p]  # drop empties

        pred = remove_adjacent_duplicates(pred)
        return pred
    except Exception:
        return []


# # Calculate evaluation metrics


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

def token_levenshtein(a, b):
    """Edit distance on token sequences (lists), like Levenshtein but for lists."""
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
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[n][m]

def cal_index(index, label, pred):
    # Normalize
    label = [normalize_place(x) for x in label if x]
    pred  = [normalize_place(x) for x in pred  if x]

    # (optional) remove adjacent dups post-normalization
    def dedup_adj(xs):
        out = []
        for x in xs:
            if not out or out[-1] != x:
                out.append(x)
        return out
    label = dedup_adj(label)
    pred  = dedup_adj(pred)


    # === LN: number of label places.   PN: number of predicted places ===
    LN, PN = len(label), len(pred)

    # === TP: True Positive ===
    pairs = best_matches(label, pred, thr=0.80)     # ← replace greedy loop
    TP = len(pairs)

    # === Calculate Kendall_tau ===
    if TP >= 2:
        label_idx = [i for (i, _, _) in pairs]
        pred_idx  = [j for (_, j, _) in pairs]
        kt = stats.kendalltau(label_idx, pred_idx, nan_policy='omit')[0]
        Kendall_tau = 0.0 if (kt is None or np.isnan(kt)) else float(kt)
    else:
        Kendall_tau = 0.0

    # Keep your original distance/similarity behavior
    sequences_l = label
    sequences_p = pred

    # Optional: align matched slots so edit distance gives zero cost on those
    # (not required for τ, but preserves your original intent)
    sequences_p = list(sequences_p)
    for (i, j, _) in pairs:
        sequences_p[j] = sequences_l[i]

    # === edit silimarity ===
    dis = token_levenshtein(sequences_l, sequences_p)
    denom = (len(sequences_l) + len(sequences_p))
    edit_simlar = 0.0 if denom == 0 else 1 - dis / denom

    return index, TP, LN, PN, Kendall_tau, dis, edit_simlar



def cal_index_df(merge):
    index = merge.apply(lambda x: cal_index(x['id'] ,x.label, x.pred), axis=1, result_type='expand')
    index.columns = ['id','TP', 'LN', 'PN', 'Kendall_tau', 'dis', 'edit_simlar']
    index.set_index('id', inplace=True)
    return index


def cal_index_all(merge):
    index = merge.apply(lambda x: cal_index(x['id'], x.label, x.pred), axis=1, result_type='expand')
    index.columns = ['id', 'TP', 'LN', 'PN', 'Kendall_tau', 'dis', 'edit_simlar']

    TP_sum, PN_sum, LN_sum = index.TP.sum(), index.PN.sum(), index.LN.sum()
    precision = 0.0 if PN_sum == 0 else TP_sum / PN_sum
    recall    = 0.0 if LN_sum == 0 else TP_sum / LN_sum
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    edit_simlar = index.edit_simlar.mean()
    kendall_tau = 0.0 if TP_sum == 0 else float((index.Kendall_tau * index.TP).sum() / TP_sum)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'kendall_tau': kendall_tau, 'edit_simlar': edit_simlar}

def parse_llm_json(raw_text: str):
    """Parse LLM output that may include prose or ```json fences; return a Python obj."""
    if not isinstance(raw_text, str):
        return raw_text
    m = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.I|re.S)
    if m:
        raw_text = m.group(1)
    obj = json_repair.loads(raw_text)
    if isinstance(obj, str):
        try:
            obj = json_repair.loads(obj)
        except Exception:
            pass
    return obj



if __name__ == "__main__":
    # 1. 采样—归纳流程

    ## 轨迹抽取的多样化知识
    ### 3.2.1 文本到轨迹的转换模型
    tran_model = '''
    在叙事学中，文本的叙事空间通常被划分为三个垂直层级：文本空间、拓扑空间与时空空间。
    文本空间指线性的叙事表达形式，是传递叙事的直接载体；拓扑空间表示叙事中的静态空间实体，
    如城市或乡村，在文本中以地点描述的形式出现；时空空间则包含动态的时空要素，如事件与移动。

    基于叙事空间理论，从叙事文本中抽取轨迹，实质上是将轨迹信息从“文本空间”转化到“时空空间”。
    然而，由于文本空间的线性结构与选择性表达，会对拓扑与时空信息进行压缩，使得直接完成这种
    转换变得具有挑战。受现有将过程拆解的轨迹抽取方法启发，我们将该转换分解为如下子转换：

    • 元素转换：将文本空间中的线性文本表述，转换为拓扑空间中的静态地理实体描述。
    • 行动者中心转换：把文本空间中的行动者与拓扑空间的静态地理实体建立关联。
    • 序关系转换：在时空空间中刻画行动者在不同地理实体之间的移动顺序。
    '''

    ### 3.2.2 简化的轨迹数据模型
    data_model = '''```json
[
    {
        "Actor": XXX,
        "Track": [{"Place": XXX}, ...]
    },
    ...
]```'''

    ### 3.2.3 人工标注的文本—轨迹配对
    def random_sample():
        text_df = pd.DataFrame(jsonlines.open("datasets/travel_blogs/texts.txt"))
        label_df = pd.DataFrame(jsonlines.open("datasets/travel_blogs/labels.txt"))
        data = pd.merge(text_df, label_df, on='id')
        sample = data.sample(1)
        return sample.text, sample.label

    ### 3.3 来自 LLM 的采样
    res = []
    for i in range(3):
        text, traj = random_sample()
        prompt = '''请基于“转换模型”和“文本—轨迹示例”，提出一种用于从文本中抽取轨迹的【逐步推理路径】与【约束条件】。'''
        input_for_sample = f'''
## 转换模型（Transformation Model）：{tran_model}
## 文本—轨迹示例（Text-trajectory Example）：
## 文本（Text）：{text}
## 轨迹（Trajectory）：{traj}'''
        res.append(chatbot(prompt, input_for_sample))

    ### 3.3 将归纳得到的“推理路径”和“约束条件”与提示模板组合
    #### 下方为归纳得到的“推理路径（Reasoning Path）”与“约束条件（Constraints）”示例

    rea_path = '''
1. 识别文本中的所有地点（Places）与时间（Times）实体。
2. 判断每个地点是否为行动者实际到访的地点。
3. 按照行动者的到访顺序构建其轨迹。
'''
    constraints = '''
- 确保每一个轨迹点仅对应单一地点；若包含多个地点，请拆分为多个轨迹点。
- 确保轨迹点均为行动者实际到访的地点；若并非到访，请将其从轨迹中移除。
- 确保轨迹点按到访顺序排列；若顺序不正确，请进行重排。
'''

    prompt_template = f'''
# 任务
- 请从以下文本中抽取行动者的轨迹。

## 步骤
{rea_path}

## 反思与约束
{constraints}

## 输出格式
{data_model}
'''


    # 2. Load data and (optionally) extract trajectory
    text_jsonl = "datasets/travel_blogs/texts.txt"

    if RUN_INFERENCE:
        text_df = pd.DataFrame(jsonlines.open(text_jsonl))
        pred_df = pd.DataFrame(columns=['id', 'pred'])

        for id, text in text_df[['id', 'text']].values:
            try:
                output_raw = chatbot(prompt_template, text)
                obj = parse_llm_json(output_raw)

                # Normalize to list-of-actors
                if isinstance(obj, dict):
                    actors = [obj]
                elif isinstance(obj, list):
                    actors = obj
                else:
                    raise ValueError("Unsupported LLM output root type.")

                actor_block = next((a for a in actors if isinstance(a, dict) and 'Track' in a), None)
                if actor_block is None:
                    raise ValueError("No actor block with 'Track' found.")

                actor = actor_block.get('Actor', 'Actor')
                track = actor_block.get('Track', [])

                norm_track = []
                for t in track:
                    if isinstance(t, dict) and 'Place' in t and isinstance(t['Place'], str):
                        norm_track.append({'Place': t['Place']})
                    elif isinstance(t, str):
                        norm_track.append({'Place': t})
                    elif isinstance(t, dict) and len(t) == 1:
                        val = next(iter(t.values()))
                        if isinstance(val, str):
                            norm_track.append({'Place': val})

                result = [{"Actor": actor, "Track": norm_track}]
                pred_df = pred_df.append({'id': id, 'pred': result}, ignore_index=True)

            except Exception as e:
                print(f"Error: {id}")
                print(e)
            
            time.sleep(1)

        pred_df.to_json(PREDS_PATH, orient='records', lines=True)
    else:
        # Sanity check: ensure preds file exists when skipping inference
        if not os.path.exists(PREDS_PATH):
            raise FileNotFoundError(f"Preds file not found at {PREDS_PATH}. Set RUN_INFERENCE=True to generate it.")

    # 3. Load data and calculate evaluation metrics (use normalization helpers)
    label_df = get_label(LABELS_PATH)   # -> columns: id, label (list[str])
    pred_df  = get_pred(PREDS_PATH)     # -> columns: id, pred  (list[str])

    # Keep only needed columns, then merge by id
    merge = pd.merge(label_df[['id', 'label']], pred_df[['id', 'pred']], on='id', how='inner')

    # Per-id metrics table
    index = cal_index_df(merge)
    print(index)

    # Overall summary
    summary = cal_index_all(merge)
    print("\nOverall:")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")