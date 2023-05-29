'''
This script preprocesses the debates (lemmatization, postagging, sentence splitting...) and puts them in the format needed for the coreference solver.
It is partly adapted from https://github.com/samlee946/utd-codi-crac2022/blob/main/data/Convert.ipynb
'''

import json
import os
from transformers import BertTokenizer
import nltk
import pandas as pd
import re
import string
import spacy
import argparse
from utils import load_iq2

from nltk.corpus import stopwords


sw = set(stopwords.words("english")) | set(string.punctuation) | set([''])
reporting_verbs = ['ask', 'know', 'remember', 'say', 'see', 'add', 'admit', 'agree', 'announce', 'answer', 'argue',
                   'boast', 'claim', 'comment', 'complain', 'confirm', 'consider', 'deny', 'doubt', 'estimate', 'explain', 'fear', 'feel', 'insist', 'mention', 'observe', 'persuade', 'propose', 'remark',
                   'remember', 'repeat', 'reply', 'report', 'reveal', 'say', 'state', 'suggest', 'suppose', 'tell', 'think',
                        'understand', 'warn', 'decide', 'expect', 'guarantee', 'hope', 'promise', 'swear', 'threaten',
                   'advise', 'beg', 'demand', 'insist', 'prefer', 'propose', 'recommend', 'request', 'suggest',
                   'decide', 'describe', 'discover', 'discuss', 'explain', 'forget', 'guess', 'imagine', 'know', 'learn', 'realise',
                   'remember', 'reveal', 'say', 'see', 'suggest', 'teach', 'tell', 'think', 'understand', 'wonder', 'advise', 'ask',
                   'beg', 'command', 'forbid', 'instruct', 'invite', 'teach', 'tell', 'warn']
removal_list = set(['know', 'mean', 'think', 'believe', 'say', 'see'] + reporting_verbs)






def remove_nonsense_and_deal(l, anaphor_idx=None):
    new_list = []
    new_anaphor_idx = -1
    for idx, word in enumerate(l):
        lowered_word = word.lower().replace("'", "")
        if lowered_word in ['nt', "n't"] and len(new_list):
            new_list[-1] = new_list[-1] + "n't"
        elif lowered_word in connect_list and len(new_list):
            new_list[-1] = new_list[-1] + "'" + word.replace("'", "")
        elif lowered_word in nonsense_list:
            continue
        else:
            if anaphor_idx is not None and anaphor_idx == idx:
                new_anaphor_idx = len(new_list)
            new_list.append(word)
    if anaphor_idx is not None:
        return new_list, new_anaphor_idx
    else:
        return new_list


cached = {}


def only_remove_nonsense(l, anaphor_idx=None):
    if (tuple(l), anaphor_idx) in cached:
        new_list, new_anaphor_idx = cached[(tuple(l), anaphor_idx)]
        if anaphor_idx is not None:
            return new_list, new_anaphor_idx
        else:
            return new_list
    new_list = []
    new_anaphor_idx = -1
    for idx, word in enumerate(l):
        lowered_word = word.lower().replace("'", "")
        if len(new_list) > 0 and lowered_word == 'na' and new_list[-1].lower() == 'gon':
            new_list[-1] = new_list[-1] + 'na'
        elif lowered_word in nonsense_list:
            continue
        else:
            if anaphor_idx is not None and anaphor_idx == idx:
                new_anaphor_idx = len(new_list)
            if lowered_word in ['s', 've']:
                word = f"'{word}"
            new_list.append(word)
    cached[(tuple(l), anaphor_idx)] = (new_list, new_anaphor_idx)
    if anaphor_idx is not None:
        return new_list, new_anaphor_idx
    else:
        return new_list


def get_num_of_words_for_rule_based(l):
    doc = nlp(l)
    cnt = 0
    for tok in doc:
        if tok.pos_ in ['VERB', 'NOUN', 'PROPN'] and tok.lemma_ not in removal_list:
            cnt += 1
    return cnt


def get_first_verb(tok):
    if tok.pos_ == 'VERB':
        return tok.i
    elif tok.i == tok.head.i:
        return -1
    else:
        return get_first_verb(tok.head)


def get_dep_parsing(l, anaphor_idx=None):
    doc = nlp(l)
    prev_verb_set = set()
    flag = True
    #     print(doc)
    for tok in doc:
        #         print(tok.i, tok, tok.pos_, tok.lemma_, tok.dep_, tok.head.i, get_first_verb(tok))
        if anaphor_idx is not None:
            if tok.i < anaphor_idx:
                prev_verb_set |= set([get_first_verb(tok)])
            elif tok.i == anaphor_idx:
                anaphor_verb = get_first_verb(tok)
                if len(prev_verb_set - set([-1])) == 1 and list(prev_verb_set)[
                    0] == anaphor_verb and anaphor_verb != -1:
                    flag = False
    #                     print(doc, anaphor_idx, doc[anaphor_idx], prev_verb_set, anaphor_verb)
    #     spacy.displacy.serve(doc)
    return flag


ALL_PRONOUNS = set(
    ['all', 'another', 'any', 'anybody', 'anyone', 'anything', 'as', 'aught', 'both', 'each', 'each other', 'either',
     'enough', 'everybody', 'everyone', 'everything', 'few', 'he', 'her', 'hers', 'herself', 'him', 'himself', 'his',
     'I', 'idem', 'it', 'its', 'itself', 'many', 'me', 'mine', 'most', 'my', 'myself', 'naught', 'neither', 'no one',
     'nobody', 'none', 'nothing', 'nought', 'one', 'one another', 'other', 'others', 'ought', 'our', 'ours', 'ourself',
     'ourselves', 'several', 'she', 'some', 'somebody', 'someone', 'something', 'somewhat', 'such', 'suchlike', 'that',
     'thee', 'their', 'theirs', 'theirself', 'theirselves', 'them', 'themself', 'themselves', 'there', 'these', 'they',
     'thine', 'this', 'those', 'thou', 'thy', 'thyself', 'us', 'we', 'what', 'whatever', 'whatnot', 'whatsoever',
     'whence', 'where', 'whereby', 'wherefrom', 'wherein', 'whereinto', 'whereof', 'whereon', 'wherever', 'wheresoever',
     'whereto', 'whereunto', 'wherewith', 'wherewithal', 'whether', 'which', 'whichever', 'whichsoever', 'who',
     'whoever', 'whom', 'whomever', 'whomso', 'whomsoever', 'whose', 'whosever', 'whosesoever', 'whoso', 'whosoever',
     'ye', 'yon', 'yonder', 'you', 'your', 'yours', 'yourself', 'yourselves'])

nonsense_list = ['yeah', 'okay', 'ok', 'uh', 'right', 'so', 'hmm', 'well', 'um', 'oh', 'mm', 'yep', 'hi', 'ah',
                 'whoops', 'alright', 'shhhh', 'yes', 'ay', 'hello', 'aww', 'alas', 'ye', 'aye', 'uh-huh', 'huh', 'wow', 'www', 'no', 'and ', 'but', 'again', 'wonderful', 'exactly', 'absolutely', 'actually',
                 'sure thanks', 'awesome', 'gosh', 'ooops']
connect_list = ["ve", "na", "re", "m", "s", "d"]


def split_into_segments(all_subtokens, all_speakers, subtoken_map, max_seg_len, constraints1, constraints2,
                           tokenizer):
    '''
    Add subtokens, subtoken_map, info for each segment; add CLS, SEP in the segment subtokens
        Input document_state: tokens, subtokens, token_end, sentence_end, utterance_end, subtoken_map, info
    '''
    curr_idx = 0  # Index for subtokens
    prev_token_idx = 0
    segments = []
    speakers = []
    segment_subtoken_map = []
    final_sentence_map = []
    while curr_idx < len(all_subtokens):
        # Try to split at a sentence end point
        end_idx = min(curr_idx + max_seg_len - 1 - 2, len(all_subtokens) - 1)
        while end_idx >= curr_idx and not constraints1[end_idx]:
            end_idx -= 1
        if end_idx < curr_idx:
            print('no sentence end found; split at token end')
            # If no sentence end point, try to split at token end point
            end_idx = min(curr_idx + max_seg_len - 1 - 2, len(all_subtokens) - 1)
            while end_idx >= curr_idx and not constraints2[end_idx]:
                end_idx -= 1
            if end_idx < curr_idx:
                print('Cannot split valid segment: no sentence end or token end')

        segment = [tokenizer.cls_token] + all_subtokens[curr_idx: end_idx + 1] + [tokenizer.sep_token]
        speaker_info = ['[SPL]'] + all_speakers[curr_idx: end_idx + 1] + ['[SPL]']

        segments.append(segment)
        speakers.append(speaker_info)

        subtoken_map_h = subtoken_map[curr_idx: end_idx + 1]
        segment_subtoken_map.append([prev_token_idx] + subtoken_map + [subtoken_map[-1]])

        curr_idx = end_idx + 1
        prev_token_idx = subtoken_map[-1]

    return segments, speakers


def get_conv_data(data, convname):
    conv_instances = data[data["ConvID"] == convname]
    conv_utterances = conv_instances["Utterance"]
    speakers_by_utterance = list(conv_instances["Speaker"].values)
    speakertypes_by_utterance = list(conv_instances["SpeakerType"].values)
    conv_tokens = []
    conv_sentences = []
    all_utterances_tokennums = []
    token_num = 0
    for utt in conv_utterances:
        utt_sent_tokennums = []
        doc = nlp(utt)
        sentence_split = list(doc.sents)
        for sentence in sentence_split:
            sent_tokennums = []
            tokens_in_sent = nltk.word_tokenize(sentence.text)
            conv_tokens.extend(tokens_in_sent)
            for tok in tokens_in_sent:
                sent_tokennums.append(token_num)
                token_num += 1
            utt_sent_tokennums.append(sent_tokennums)
        all_utterances_tokennums.append(utt_sent_tokennums)
    return conv_tokens, all_utterances_tokennums, speakers_by_utterance, speakertypes_by_utterance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data_for_coref/")
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_md")
    data = load_iq2()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased") # as in the original coreference solver
    max_seg_len = 512
    all_data = []
    for convname in data["ConvID"].unique():
        print(convname)
        conv_tokens, all_utterances_tokennums,  speakers_by_utterance, speakertypes_by_utterance = get_conv_data(data, convname)
        one_debate_nf = {'doc_key': convname, "tokens": conv_tokens, "tokens_by_sentence_and_utterance": all_utterances_tokennums,
                         "speakers_by_utterance":  speakers_by_utterance, "speakertypes_by_utterance": speakertypes_by_utterance}
        for empty_key in ['ner', 'constituents', 'clusters', 'preceding_5_words', 'following_5_words',
                          'candidate_entity_spans', 'pronouns']:
            one_debate_nf[empty_key] = []

        all_subtokens = []
        subtoken_map = []
        sentence_end = []
        token_end = []
        all_speakers = []
        word_idx = -1
        sentence_map = []
        utterance_spans = []
        sent_idx = -1
        subtk_idx = -1
        for utt, utt_speaker in zip(one_debate_nf['tokens_by_sentence_and_utterance'],
                                    one_debate_nf['speakers_by_utterance']):
            utt_start = 0 if subtk_idx < 0 else subtk_idx
            for sent in utt:
                sent_idx += 1
                for idx_in_sent, word_num in enumerate(sent):
                    word_idx += 1
                    word = one_debate_nf['tokens'][word_num]
                    subtokens = tokenizer.tokenize(word)
                    token_end += [False] * (len(subtokens) - 1) + [True]
                    for idx, subtoken in enumerate(subtokens):
                        subtk_idx += 1
                        all_subtokens.append(subtoken)
                        if idx_in_sent == len(sent) - 1:
                            sentence_end.append(True)
                        else:
                            sentence_end.append(False)
                        subtoken_map.append(word_idx)
                        all_speakers.append(utt_speaker)
                        sentence_map.append(sent_idx)
                one_debate_nf['subtoken_map'] = subtoken_map
            utt_end = subtk_idx
            utterance_spans.append((utt_start, utt_end, "utterance"))

        constraits1 = sentence_end
        constraits2 = token_end
        one_debate_nf['utterance_span'] = utterance_spans
        one_debate_nf['clusters_dd'] = []
        one_debate_nf['gold_anaphors'] = []

        segments, speakers = split_into_segments(all_subtokens, all_speakers, subtoken_map, max_seg_len,
                                                    constraits1, constraits2, tokenizer)
        one_debate_nf['sentences'] = segments
        one_debate_nf['speakers'] = speakers
        one_debate_nf['sentence_map'] = sentence_map


        # fill in pronouns
        for x in range(len(subtoken_map)):
            for y in range(x, min(x + 10, len(subtoken_map))):
                st = subtoken_map[x]
                ed = subtoken_map[y]
                tok = ' '.join(one_debate_nf['tokens'][st:ed + 1]).lower()
                #                 print(x, y, st, ed, tok, tok in ALL_PRONOUNS)
                if tok in ALL_PRONOUNS:
                    one_debate_nf['pronouns'].append([x, y])

        doc = one_debate_nf
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        with open(os.path.join(args.out_dir, convname + ".jsonlines"), "w") as output_file:
            output_file.write(json.dumps(doc))
            output_file.write('\n')

