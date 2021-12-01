from typing import Any, Dict, List, Optional, Tuple, Union, Iterator

import json
import os
from collections import namedtuple
import random
import colorlog
from operator import itemgetter
import argparse

from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from parlai.core.dict import DictionaryAgent
from parlai.core.worlds import create_task


PARLAI_KNOWLEDGE_SEPARATOR = '__knowledge__'
BERT_KNOWLEDGE_SEPARATOR = '_ _ knowledge _ _'

_PARLAI_PAD = '__null__'
_PARLAI_GO = '__start__'
_PARLAI_EOS = '__end__'
_PARLAI_UNK = '__unk__'


DEFAULT_TYPE_ID = 0
USER_TYPE_ID = 1
AGENT_TYPE_ID = 2


class WowDatasetReader:

    def __init__(self,
                 cache_dir: str = None) -> None:
        self._datapath = cache_dir


    def _load_and_preprocess_all(self, mode: str):
        """
        As default, it returns the following action dict:
        {
            'id': 'wizard_of_wikipedia'
            'text': chosen_topic\n # if first example in episode
                    last_apprentice_message\n # if possible
                    wizard_message # if --label-type is 'chosen_sent'
            'knowledge': title_1 sentence_1\n
                                .
                                .
                                .
                         title_m sentence_n # all knowledge available to wizard
            'labels': [title_checked sentence_checked] # default
                                    OR
                      [wizard_response] # if --label-type set to 'response'
            'label_candidates': knowledge + [no_passages_used no_passages_used]
                                           OR
                                100 response candidates  # if 'validation' or 'test'
            'chosen_topic': chosen_topic as untokenized string
            'checked_sentence': checked sentence if wizard, else None # if --include_checked_sentence
            'title': title of checked sentence # if --include_checked_sentence
            --> if not exists, then checked_sentence = title = 'no_passages_used'
            'episode_done': (Boolean) whether episode is done or not
        }
        """

        parlai_opt = self._get_parlai_opt([
            '--task', 'wizard_of_wikipedia:generator:topic_split' if 'unseen' in mode else 'wizard_of_wikipedia:generator:random_split',
            '--datatype', '{}:stream'.format(mode.split('_')[0]) if 'unseen' in mode else f'{mode}:stream',  # 'train' for shuffled data and 'train:stream' for unshuffled data
            '--datapath', self._datapath,
            # dict_XXX will not be used if we use bert tokenizer
            '--dict_lower', 'True',
            '--dict_tokenizer', 'bpe',
            '--dict_file', f"{self._datapath}/wow.dict",
            '--dict_textfields', "text,labels,chosen_topic,checked_sentence,knowledge,title",  # For retrieval mode, use "text,labels"
            # By following author's code. For retrieval mode, use 250004
            # Also, note that this is the size of bpehelper dictionary.
            # So, final dictionary can be larger than this one
            # And, don't convert special tokens to index with txt2vec method, you must use tok2ind
            '--dict_maxtokens', '30000',
            '--dict_nulltoken', _PARLAI_PAD,
            '--dict_starttoken', _PARLAI_GO,
            '--dict_endtoken', _PARLAI_EOS,
            '--dict_unktoken', _PARLAI_UNK,
            '--include_knowledge_separator', 'True',  # include speical __knowledge__ token between title and passage
            '--include_checked_sentence', 'True',
            '--label_type', 'response', # choices = ['response', 'chosen_sent']
        ])
        # As a default, world use "WizardDialogKnowledgeTeacher"
        agent = DictionaryAgent(parlai_opt)
        world = create_task(parlai_opt, agent)
        num_examples = world.num_examples()
        num_episodes = world.num_episodes()

        episodes = []
        for _ in range(num_episodes):
            examples = []
            while True:
                world.parley()
                example = world.acts[0]
                examples.append(example)
                if world.episode_done():
                    episodes.append(examples)
                    break

        return self._preprocess_episodes(episodes, mode)

    def _get_parlai_opt(self, options: List[str] = [], print_args=False):
        from parlai.scripts.build_dict import setup_args
        parser = setup_args()
        opt = parser.parse_args(options, print_args=print_args)
        return opt

    def _get_preprocessed_fname(self, mode):
        if self._datapath:
            return os.path.join(self._datapath, f'{mode}_episodes.json')
        else:
            return None

    def _preprocess_episodes(self, episodes, mode):

        colorlog.info("Preprocess wizard of wikipedia dataset")

        new_episodes = []
        for episode_num, episode in enumerate(tqdm(episodes, ncols=70)):
            new_examples = []
            for example_num, example in enumerate(episode):
                context = example['text']
                if mode == "train":
                    response = example['labels'][0]
                else:
                    response = example['eval_labels'][0]
                chosen_topic = example['chosen_topic']

                # Set up knowledge
                checked_knowledge = example['title'] + ' __knowledge__ ' + example['checked_sentence']
                knowledge_sentences = [k for k in example['knowledge'].rstrip().split('\n')]
                assert "no_passages_used __knowledge__ no_passages_used" in knowledge_sentences
                
                for idx, k in enumerate(knowledge_sentences):
                    if k == checked_knowledge:
                        break
                else:
                    # Sometimes, knowledge does not include checked_sentnece
                    idx = None
                    colorlog.warning("Knowledge does not include checked sentence.")
                if idx is None:
                    knowledge_sentences += [checked_knowledge]

                new_example = {'context': context,
                               'response': response,
                               'chosen_topic': chosen_topic,
                               'knowledge_sentences': knowledge_sentences,
                               'chosen_knowledge': checked_knowledge,
                               'episode_num': episode_num,
                               'example_num': example_num}
                new_examples.append(new_example)
            new_episodes.append(new_examples)

        return new_episodes



class WoWPassage:

    def __init__(self, title, position, sentences):
        self.id = ""  # put a dummy one
        self.title = title
        self.position = position
        self.text = sentences
        self.type = [0] * len(sentences) # put a dummy one
        self.has_answer = None
        self.history_answers_spans = None
        self.history_has_answers = None
        self.answers_spans = None


class WoWDataSample:

    def __init__(self, conv_id, turn_id, chosen_sentence):
        self.id = f'conv_{conv_id}_turn_{turn_id}'

        self.question = None

        self.question_type = None # list of 1s and 2s indicating previous turn roles
        self.history_dialog_act = None 
        self.dialog_act = 0  # put a dummy one
        self.answers = [chosen_sentence]
        
        self.ctxs = None


    def set_question(self, question, question_type):
        self.question = question
        self.question_type = question_type
        assert len(question_type) == len(question)
        self. history_dialog_act = [0] * len(question) # put a dummy one


    def add_passage(self, title, position, sentences, has_answer, chosen_sentence, history_has_answers, history_sentences):
        if self.ctxs is None:
            self.ctxs = []
        
        passage = WoWPassage(title, position, sentences)
        passage.has_answer = has_answer
        if has_answer:
            sentence_id = sentences.index(chosen_sentence)
            assert sentence_id >= 0
            passage.answers_spans = [[sentence_id, sentence_id]]
        
        passage.history_has_answers = history_has_answers
        history_answers_spans = []
        for i, hist_has_ans in enumerate(history_has_answers):
            if not hist_has_ans:
                history_answers_spans += [None]
            else:
                sentence_id = sentences.index(history_sentences[i])
                assert sentence_id >= 0
                history_answers_spans += [[sentence_id, sentence_id]]
        passage.history_answers_spans = history_answers_spans
        
        self.ctxs += [vars(passage)]


class WoWDialog():
    
    def __init__(self, turns):

        self.dialid = turns[0]['episode_num']
        self.chosen_topic = turns[0]['chosen_topic']
        self.utterances = turns
    
    def read_samples(self):

        samples = []

        prev_turns = []
        prev_type = []
        prev_sentences = []
        prev_passages = []

        perc_found_prev_answers = []

        for i, turn in enumerate(self.utterances):

            assert i == turn['example_num']
            assert self.dialid == turn['episode_num']
            
            user_turn = turn['context']
            agent_turn = turn['response']

            prev_passages += [None]
            prev_sentences += [None]
            prev_type += [USER_TYPE_ID]
            if i == 0:
                # the first user turn includes topic at the beginning
                topic = user_turn[:len(self.chosen_topic)]
                assert topic == self.chosen_topic
                # original topic and text separator is '\n'
                assert user_turn == topic or user_turn[len(self.chosen_topic)] == '\n'
                text = user_turn[len(self.chosen_topic):]
                if text:
                    prev_turns += [f"{topic} <user> {text.strip()}"]
                else:
                    prev_turns += [f"{topic}"]
            else:
                prev_turns += [f"<user> {user_turn.strip()}"]


            chosen_passage, chosen_sentence = turn['chosen_knowledge'].split(' __knowledge__ ')
            assert turn['chosen_knowledge'] in turn['knowledge_sentences']
            # make data sample
            sample = WoWDataSample(self.dialid, i, chosen_sentence)
            sample.set_question(list(reversed(prev_turns)), list(reversed(prev_type)))


            title2sentences = defaultdict(list)
            all_k_sents = turn["knowledge_sentences"]
            prev_title = None
            titles = []
            for j, sent in enumerate(all_k_sents):
                try:
                    title, k_sent = sent.split(' __knowledge__ ')
                except:
                    continue

                title2sentences[title] += [k_sent]
                prev_title = title
                if title not in titles:
                    titles += [title]


            total_has_answer = 0
            total_prev_has_answer = [0] * len(prev_turns)
            for position, title in enumerate(titles):

                sentences = title2sentences[title]

                has_answer = (chosen_passage == title and chosen_sentence in sentences)
                total_has_answer += int(has_answer)
                    
                prev_has_answers = [(prev_p == title and prev_sent in sentences) for prev_p, prev_sent in zip(prev_passages, prev_sentences)]
                total_prev_has_answer = [total_prev_has_answer[i]+int(ans) for i, ans in enumerate(prev_has_answers)]

                sample.add_passage(title,
                                   position,
                                   sentences,
                                   has_answer,
                                   chosen_sentence,
                                   list(reversed(prev_has_answers)), 
                                   list(reversed(prev_sentences)))

            assert total_has_answer == 1

            n_found_prev_answers = sum([total_prev_has_answer[i] for i, ans in enumerate(total_prev_has_answer) if prev_type[i] == AGENT_TYPE_ID])
            n_prev_agent_turns = len([t for t in prev_type if t == AGENT_TYPE_ID])
            perc_found_prev_answers += [n_found_prev_answers*1.0/(n_prev_agent_turns+1e-9)]

            samples += [sample]

            prev_passages += [chosen_passage]
            prev_sentences += [chosen_sentence]
            prev_type += [AGENT_TYPE_ID]
            prev_turns += [f"<agent> {agent_turn.strip()}"]

        return samples, perc_found_prev_answers


def write_examples(dialogues, outdir, split):
    dialogues = [WoWDialog(d) for d in dialogues]
    qas = []
    perc_found_prev_answers_list = []
    for dial in dialogues:
        samples, perc_found_prev_answers = dial.read_samples()
        perc_found_prev_answers_list += perc_found_prev_answers
        qas += [vars(sample) for sample in samples]
            
    print(f'{split} examples: {len(qas)}')
    print(f'percentage of previous answers can be found in current candidate passages: {sum(perc_found_prev_answers_list)/len(perc_found_prev_answers_list)}')
    with open(os.path.join(outdir, f'{split}.json'), 'w', encoding="utf-8") as fout:
        fout.write(json.dumps(qas, indent=4))


def main(args):

    reader = WowDatasetReader(args.cache_dir)
    dialogues = reader._load_and_preprocess_all('train')
    write_examples(dialogues, args.cache_dir, 'train')
    
    dialogues = reader._load_and_preprocess_all('valid')
    write_examples(dialogues, args.cache_dir, 'dev')

    dialogues = reader._load_and_preprocess_all('valid_unseen')
    write_examples(dialogues, args.cache_dir, 'dev_unseen')
    
    dialogues = reader._load_and_preprocess_all('test')
    write_examples(dialogues, args.cache_dir, 'test')
    
    dialogues = reader._load_and_preprocess_all('test_unseen')
    write_examples(dialogues, args.cache_dir, 'test_unseen')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cache_dir",
        type=str,
        default='',
        help="directory path of the output data",
    )
    args = parser.parse_args()


    main(args)
