import time
import json
import os
import re
import sys
import random
from collections import defaultdict, Counter
import argparse
import nltk
from transformers import BertTokenizer
import string

DEFAULT_TYPE_ID = 0
USER_TYPE_ID = 1
AGENT_TYPE_ID = 2

DA_ID_MAP = {"respond_solution": 0, "query_condition": 1, "respond_solution_positive": 2, "respond_solution_negative": 3,
             "query_solution": 4, "response_negative": 5, "response_positive": 6}

def _get_answers_rc(references, spans, doc_text, spid2passagelocation):
    """Obtain the grounding annotation for a given dialogue turn"""
    if not references:
        return []
    start, end = -1, -1
    ls_sp = []
    secid2count = defaultdict(int)
    for ele in references:
        sp_id = ele["sp_id"]
        secid2count[spid2passagelocation[sp_id][0]] += 1

    secid = sorted(secid2count.items(), key=lambda item: -item[1])[0][0]

    start_spid = None
    end_spid = None
    for ele in references:
        sp_id = ele["sp_id"]

        start_sp, end_sp = spans[sp_id]["start_sp"], spans[sp_id]["end_sp"]
        if start == -1 or start > start_sp:
            start = start_sp
            start_spid = sp_id
        if end < end_sp:
            end = end_sp
            end_spid = sp_id
        ls_sp.append(doc_text[start_sp:end_sp])
    answer = doc_text[start:end]
    ans_spids = [str(id) for id in range(int(start_spid),int(end_spid)+1)] if (start_spid and end_spid) else []
    return [' '.join(answer.strip().split())], secid, ans_spids

def _load_doc_data_rc(filepath):
    doc_filepath = os.path.join(os.path.dirname(filepath), "doc2dial_doc.json")
    with open(doc_filepath, encoding="utf-8") as f:
        data = json.load(f)["doc_data"]
    return data

def _get_section_info(doc):
    # assumes that doc spans and sections (passages) are put in order in the doc file
    id2text = defaultdict(list)
    id2spid = defaultdict(list)
    id2ptitles = defaultdict(list)
    ids_in_order = []
    for sp_id in doc["spans"]:
        title = doc["spans"][sp_id]['title']
        text_sp = doc["spans"][sp_id]["text_sp"].strip()
        
        id2ptitles[title] += [ptitle["text"] for ptitle in doc["spans"][sp_id]["parent_titles"]]
        id2text[title] += [text_sp]
        id2spid[title] += [sp_id]
        if title not in ids_in_order:
            ids_in_order += [title]
    secid2text = {}
    secid2spid = {}
    for id in id2text:
        texts = [k for k in id2text[id]]
        prefix = f'<text>'
        for ptitle in id2ptitles[id]:
            prefix = f'<parent_title> {ptitle} {prefix}'
        texts = [prefix] + texts
        secid2text[id] = texts
        secid2spid[id] = [-1] + [k for k in id2spid[id]]
    return secid2text, secid2spid, ids_in_order

def _get_spid2type(doc):
    spid2type = {}
    cur_type = 1
    prev_id_sec = None
    for i, sp_id in enumerate(doc["spans"]):
        id_sec = doc["spans"][sp_id]["id_sec"]
        if prev_id_sec and id_sec != prev_id_sec:
            cur_type = 1 - cur_type
        spid2type[sp_id] = cur_type
        prev_id_sec = id_sec
        
    return spid2type

def _get_passage_type_ids(args, spids, spid2type):
    type_id_list = []
    for spid in spids:
        if spid  == -1:
            type_id_list += [DEFAULT_TYPE_ID]
        else:
            sp_type = spid2type[spid]
            type_id_list += [sp_type]
    return type_id_list


def _process_sections(args, doc_id, secid2text, secid2spid, ids_in_order, spid2type):
    ctxs = []
    spid2passagelocation = {}
    secid2position = {}
    for i, secid in enumerate(ids_in_order):
        assert len(secid2text[secid]) == len(secid2spid[secid])
        secid2position[secid] = i
        type_id_list = _get_passage_type_ids(args, secid2spid[secid], spid2type)
        ctx = {'id': secid, "title": doc_id, "position": i, "text": secid2text[secid], "type": type_id_list, "has_answer": False}
        for j, spid in enumerate(secid2spid[secid]):
            if spid == -1:
                continue
            spid2passagelocation[spid] = (secid, j)
        ctxs += [ctx]
    return ctxs, spid2passagelocation, secid2position


def process_doc(args, doc, doc_id):
    secid2text, secid2spid, ids_in_order = _get_section_info(doc)
    spid2type = _get_spid2type(doc)
    ctxs, spid2passagelocation, secid2position = _process_sections(args, doc_id, secid2text, secid2spid, ids_in_order, spid2type)
    return ctxs, spid2passagelocation, secid2position


def process_ctxs_dict(args, ctxs, idx, turnid2ans_spids, spid2passagelocation, secid2position, docid, ans_secid):
    new_ctxs = [dict(ctx) for ctx in ctxs]
    for c in new_ctxs:
        c['type'] = list(c['type'])  
    history_answers_spans = [[[] for i in range(idx+1)] for j in range(len(new_ctxs))]

    for tid in sorted(turnid2ans_spids.keys(), reverse=True):
        for spid in turnid2ans_spids[tid]:
            secid, type_idx = spid2passagelocation[spid] 
            history_answers_spans[secid2position[secid]][idx - tid] += [type_idx]

    for i, c in enumerate(new_ctxs):
        if c['id'] == ans_secid:
            c["has_answer"] = True
        new_ctxs[i]['history_answers_spans'] = [[sorted(ans_sp)[0], sorted(ans_sp)[-1]] if len(ans_sp)>0 else None for ans_sp in history_answers_spans[i]]
        new_ctxs[i]['history_has_answers'] = [True if len(ans_sp)>0 else False for ans_sp in history_answers_spans[i]]
    return new_ctxs

def get_question_type_ids(args, question):
    question_type_id_list = []
    for turn in question:
        if turn.startswith('<user>'):
            question_type_id_list += [USER_TYPE_ID]
        else:
            question_type_id_list += [AGENT_TYPE_ID]
    return question_type_id_list

def find_answers_spans(cur_ctxs, ans_spids, spid2passagelocation):
    ans_string = ''
    for ctx in cur_ctxs:
        if ctx['has_answer']:
            indices = []
            for spid in ans_spids:
                indices += [spid2passagelocation[spid][1]]
            indices = sorted(indices)
            ctx['answers_spans'] = [(indices[0], indices[-1])]
            ans_string = ' '.join(ctx['text'][indices[0]:indices[-1]+1])
        else:
            ctx['answers_spans'] = None
    return cur_ctxs, ans_string


def update_turnid2ans_spids(args, idx, turn, doc, turnid2ans_spids, spid2passagelocation):
    _, _, ans_spids = _get_answers_rc(turn["references"], doc["spans"], doc["doc_text"], spid2passagelocation)
    turnid2ans_spids[idx] = list(ans_spids)
    return turnid2ans_spids
    

def is_valid_turn(idx, dial_turns, turn):
    return turn["role"] != "agent" and idx + 1 < len(dial_turns) and dial_turns[idx + 1]["role"] == "agent"


def main(args):
    dtype = args.dtype
    filepath = args.filepath.format(dtype)
    if dtype == 'validation':
        dtype = 'dev'
    outfile = args.outfile.format(dtype)


    doc_data = _load_doc_data_rc(filepath)
    qas = []
    with open(filepath, encoding="utf-8") as f:
        dial_data = json.load(f)["dial_data"]
        for domain, d_doc_dials in dial_data.items():
            for doc_id, dials in d_doc_dials.items():
                doc = doc_data[domain][doc_id]
                ctxs, spid2passagelocation, secid2position = process_doc(args, doc, doc_id)

                for dial in dials:
                    all_prev_utterances = []
                    all_prev_dialacts = []
                    turnid2ans_spids = {}
                    dial_turns = dial["turns"]
                    for idx, turn in enumerate(dial_turns):
                        all_prev_utterances.append("<{}>: {}".format(turn["role"], turn["utterance"]))
                        all_prev_dialacts.append(DA_ID_MAP[turn["da"]])

                        id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"]) 

                        turnid2ans_spids = update_turnid2ans_spids(args, idx, turn, doc, turnid2ans_spids, spid2passagelocation)
                        
                        if not is_valid_turn(idx, dial_turns, turn):
                            continue
                        turn_to_predict = dial_turns[idx + 1]
    
                        
                        question = list(reversed(all_prev_utterances))
                        dialog_act = list(reversed(all_prev_dialacts))
                        answers, ans_secid, ans_spids = _get_answers_rc(turn_to_predict["references"], doc["spans"], doc["doc_text"], spid2passagelocation)
                        cur_ctxs = process_ctxs_dict(args, ctxs, idx, turnid2ans_spids, spid2passagelocation, secid2position, doc_id, ans_secid)
                        cur_ctxs, ans_string = find_answers_spans(cur_ctxs, ans_spids, spid2passagelocation)
                        question_type_id_list = get_question_type_ids(args, question)
                        qa = {
                            "id": id_,
                            "question": question,
                            "question_type": question_type_id_list,
                            "history_dialog_act": dialog_act,
                            "dialog_act": DA_ID_MAP[turn_to_predict["da"]],
                            "answers": answers,
                            "ctxs": cur_ctxs,
                            "domain": domain,
                        }
                        qas += [qa]
    with open(outfile, 'w', encoding="utf-8") as fout:
        fout.write(json.dumps(qas, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dtype",
        required=True,
        type=str,
        help="train or validation or test",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default='./your_parent_dir/dialdoc/raw_data/doc2dial_dial_{}.json',
        help="file of the input dial data",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default='',
        help="file of the output data",
    )

    args = parser.parse_args()


    main(args)


