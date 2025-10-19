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


client = OpenAI(
# Fill in your Zhipu API Key here. You can apply for an API Key at https://www.zhipuai.cn/en/
    api_key="...", 
    base_url="https://open.bigmodel.cn/api/paas/v4/"
) 

def chatbot(prompt,text):
    completion = client.chat.completions.create(
        model="glm-4",
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
        actor_counts = dfs['Actor'].value_counts()
        most_common_actor = actor_counts.idxmax()
        pred = dfs[dfs['Actor'] == most_common_actor].reset_index(drop=True).Place.to_list()
        pred = remove_adjacent_duplicates(pred)
        return pred
    except:
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

def cal_index(index,label,pred):
    TP = 0
    LN, PN = len(label), len(pred)
    # sequences is used to calculate the edit distance by replacing the matched elements in pred with the corresponding elements in label
    sequences_l = label.copy()
    sequences_p = pred.copy()
    # match_flags is used to mark whether the elements in pred have been matched.
    match_flags = [False for j in range(len(pred))]

    # match_labels is used to store matching labels, and match_preds is used to store matching preds. Used to calculate Kendall_tau
    match_labels = [None for i in range(len(label))]
    match_preds = [None for j in range(len(pred))]
    
    for i,l in enumerate(label):
        for j,p in enumerate(pred):
            if match_flags[j]:
                continue
            # if l == p or ratio(l,p) > 0.8 or l in p or p in l:
            if l == p or ratio(l,p) > 0.8:
                TP += 1
                match_flags[j] = True
                sequences_p[j] = l
                match_labels[i] = l
                match_preds[j] = l
                break
    match_labels = [i for i in match_labels if i is not None]
    match_preds = [i for i in match_preds if i is not None]
    Kendall_tau = stats.kendalltau(match_labels, match_preds)[0]
    Kendall_tau = 0 if str(Kendall_tau) == 'nan' else Kendall_tau
    dis = distance(sequences_l, sequences_p)
    if len(sequences_l) + len(sequences_p) == 0:
        print(sequences_l, sequences_p)
    edit_simlar = 1 - dis / (len(sequences_l) + len(sequences_p))
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
    def random_sample():
        text_df = pd.DataFrame(jsonlines.open("0_Dataset/travel_blogs/texts.txt"))
        label_df = pd.DataFrame(jsonlines.open("0_Dataset/travel_blogs/labels.txt"))
        data = pd.merge(text_df, label_df,on='id')
        sample = data.sample(1)
        return sample.text, sample.label


    ### 3.3 sample from LLM

    res = []
    for i in range(3):
        text, traj = random_sample()
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


    # 2. Load data and extract trajectory
    text_jsonl = "0_Dataset/travel_blogs/texts.txt"

    text_df = pd.DataFrame(jsonlines.open(text_jsonl))

    pred_df = pd.DataFrame(columns=['id', 'pred'])

    for id, text in text_df[['id', 'text']].values:
        try:
            output = chatbot(prompt_template, text)
            output = json_repair.loads(output)
            actor = output["Actor"]
            track = output["Track"]
            place = [i["Place"] for i in track]
            result = [
                {
                    "Actor": actor,
                    "Track": [
                        {
                            "Place": place[i]
                        }
                        for i in range(len(track))
                    ]
                }
            ]
            pred_df = pred_df.append({'id': id, 'pred': result}, ignore_index=True)
        except Exception as e:
            print(f"Error: {id}")
            print(e)
    pred_df.to_json("0_Dataset/travel_blogs/preds.txt", orient='records', lines=True) 

    # 3. Load data and calculate evaluation metrics
    label_df = pd.DataFrame(jsonlines.open("0_Dataset/travel_blogs/labels.txt"))
    pred_df = pd.DataFrame(jsonlines.open("0_Dataset/travel_blogs/preds.txt"))

    merge = pd.merge(label_df, pred_df, on='id', how='inner')
    index = cal_index_df(merge)
    print(index)