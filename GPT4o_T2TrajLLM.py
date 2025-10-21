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

RUN_INFERENCE = False  # set True only when you want to re-run LLM inference
LABELS_PATH = "datasets/travel_blogs/labels.txt"
PREDS_PATH  = "datasets/travel_blogs/preds.txt"
TEXT_PATH = "datasets/travel_blogs/texts.txt"

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

def token_levenshtein(a, b):
    """Edit distance on token sequences (lists)."""
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

def normalize_place(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # strip accents
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    # lowercase
    s = s.lower().strip()
    # remove punctuation & extra spaces
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # tiny aliasing (extend as needed)
    s = s.replace("the hague", "hague")
    return s
    
# # Parse output and post-processing

def remove_adjacent_duplicates(lst):
    new_list = [lst[0]]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            new_list.append(lst[i])
    return new_list

def json2places(track_json):
    try:
        dfs = []
        for actor in track_json:
            if type(actor) == str:
                continue
            tract = [track for track in actor.get("Track",[]) if type(track) == dict]
            df = pd.DataFrame(tract)
            df['Actor'] = actor['Actor']
            dfs.append(df)
        dfs = pd.concat(dfs)
        if "Place" not in dfs.columns:
            dfs['Place'] = "XXX"
        dfs = dfs[(dfs['Place'] != 'XXX') & (dfs['Place'] != 'Unknown') & (dfs['Place'] != '')].reset_index(drop=True)
        if dfs.empty:
            return []

        # choose actor with the longest cleaned track
        best_actor = None
        best_track = []
        for actor_name, sub in dfs.groupby('Actor'):
            track = remove_adjacent_duplicates(
                [p for p in sub['Place'].astype(str).tolist() if p]
            )
            if len(track) > len(best_track):
                best_actor = actor_name
                best_track = track

        return best_track
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

def cal_index(index, label, pred):
    # Normalize first
    label = [normalize_place(x) for x in label if x]
    pred  = [normalize_place(x) for x in pred  if x]

    TP = 0
    LN, PN = len(label), len(pred)

    sequences_l = label.copy()
    sequences_p = pred.copy()

    match_flags  = [False] * PN        # whether pred[j] is already matched
    match_labels = [None] * LN         # matched token at label position i (or None)
    match_preds  = [None] * PN         # matched token at pred position j (or None)

    # One-pass greedy matching (same as before)
    for i, l in enumerate(label):
        for j, p in enumerate(pred):
            if match_flags[j]:
                continue
            # permissive after normalization
            if l == p or ratio(l, p) > 0.8 or (l and p and (l in p or p in l)):
                TP += 1
                match_flags[j]  = True
                sequences_p[j]  = l
                match_labels[i] = l
                match_preds[j]  = l
                break

    # --- Minimal fix for Kendall tau (use matched pred indices in label order) ---
    # Build token -> list of pred indices where it was matched
    idx_map = {}
    for j, tok in enumerate(match_preds):
        if tok is None:
            continue
        idx_map.setdefault(tok, []).append(j)

    # In gold (label) order, pull the next pred index for each matched token
    pred_idx_order = []
    for tok in match_labels:
        if tok is None:
            continue
        if tok in idx_map and idx_map[tok]:
            pred_idx_order.append(idx_map[tok].pop(0))

    if len(pred_idx_order) >= 2:
        Kend = stats.kendalltau(range(len(pred_idx_order)), pred_idx_order)[0]
        Kendall_tau = 0.0 if str(Kend) == 'nan' else Kend
    else:
        Kendall_tau = 0.0
    # ---------------------------------------------------------------------------
    # ------------------------------------------

    # Token-level edit distance on sequences (keep current)
    dis = token_levenshtein(sequences_l, sequences_p)
    denom = (len(sequences_l) + len(sequences_p))
    edit_simlar = 1 - dis / denom if denom > 0 else 0

    return index, TP, LN, PN, Kendall_tau, dis, edit_simlar



def cal_index_df(merge):
    index = merge.apply(lambda x: cal_index(x['id'] ,x.label, x.pred), axis=1, result_type='expand')
    index.columns = ['id','TP', 'LN', 'PN', 'Kendall_tau', 'dis', 'edit_simlar']
    index.set_index('id', inplace=True)
    return index


def cal_index_all(merge):
    index = merge.apply(lambda x: cal_index(x['id'], x.label, x.pred), axis=1, result_type='expand')
    index.columns = ['id', 'TP', 'LN', 'PN', 'Kendall_tau', 'dis', 'edit_simlar']

    precision = index.TP.sum() / index.PN.sum()
    recall = index.TP.sum() / index.LN.sum()
    f1 = 2 * precision * recall / (precision + recall)
    edit_simlar = index.edit_simlar.mean()
    kendall_tau = sum(index.Kendall_tau * index.TP) / index.TP.sum()
    return {'precision': precision, 'recall': recall, 'f1': f1, 'kendall_tau': kendall_tau, 'edit_simlar': edit_simlar}



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
        text_df = pd.DataFrame(jsonlines.open(TEXT_PATH))
        label_df = pd.DataFrame(jsonlines.open(LABELS_PATH))
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


    # 2. Load data and extract trajectory
    # text_jsonl = "datasets/travel_blogs/texts.txt"
    
    if RUN_INFERENCE:
        text_df = pd.DataFrame(jsonlines.open(TEXT_PATH))

        pred_df = pd.DataFrame(columns=['id', 'pred'])

        for id, text in text_df[['id', 'text']].values:
            try:
                output = chatbot(prompt_template, text)
                output = json_repair.loads(output)

                # --- Minimal normalization: accept either list or dict ---
                if isinstance(output, dict):
                    actors = [output]
                elif isinstance(output, list):
                    # keep only dict entries if the model mixed types
                    actors = [o for o in output if isinstance(o, dict)]
                else:
                    actors = []

                # If nothing usable, fall back to empty
                result = actors  # json2places expects a list of {"Actor":..., "Track":[{"Place":...}, ...]}

                pred_df = pred_df.append({'id': id, 'pred': result}, ignore_index=True)

            except Exception as e:
                print(f"Error: {id}")
                print(e)
            time.sleep(0.5)

        pred_df.to_json(PREDS_PATH, orient='records', lines=True, force_ascii=False) 
    
    else:
        # Sanity check: ensure preds file exists when skipping inference
        if not os.path.exists(PREDS_PATH):
            raise FileNotFoundError(f"Preds file not found at {PREDS_PATH}. Set RUN_INFERENCE=True to generate it.")

    # 3. Load data and calculate evaluation metrics (use normalization helpers)
    label_df = get_label(LABELS_PATH)   # -> columns: id, label (list[str])
    pred_df  = get_pred(PREDS_PATH)     # -> columns: id, pred  (list[str])

    merge = pd.merge(label_df, pred_df, on='id', how='inner')
    index = cal_index_df(merge)
    print(index)

    # Overall summary
    summary = cal_index_all(merge)
    print("\nOverall:")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")