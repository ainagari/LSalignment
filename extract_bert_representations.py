'''
With this script we extract and save bert representations with which we will calculate the lexicosemantic alignment measures
We also calculate similarities for the experiment where we choose the masking strategy to use.
'''

import json
from transformers import BertModel, BertTokenizer, BertConfig
import pdb
import torch
import numpy as np
from scipy.spatial.distance import cosine
import argparse
import os
import random
import pickle
from collections import defaultdict
random.seed(9)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def load_json_debate(in_dir, conv_name):
    with open(os.path.join(in_dir, conv_name)) as f:
        coref_info = json.load(f)
    return coref_info

def smart_tokenization(sentence, tokenizer, maxlen):
    cls_token, sep_token = tokenizer.cls_token, tokenizer.sep_token
    map_ori_to_bert = []
    tok_sent = [cls_token]
    incomplete = False

    for orig_token in sentence:
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize
        if len(tok_sent) + len(bert_token) >= maxlen:
            incomplete = True
            break
        tok_sent.extend(bert_token) # add to my new tokens
        if len(bert_token) > 1: # if the new token has been split
            extra = len(bert_token) - 1
            for i in range(extra):
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1)
        map_ori_to_bert.append(current_tokens_bert_idx)

    tok_sent.append(sep_token)

    return tok_sent, map_ori_to_bert, incomplete



def extract_bert_rep(berttok_sent, bert_model, bert_tokenizer, bert_target_idcs, layer):
    subword_representations = []
    with torch.no_grad():
        input_ids = torch.tensor([bert_tokenizer.convert_tokens_to_ids(berttok_sent)])
        inputs = {'input_ids': input_ids.to(device)}
        outputs = bert_model(**inputs)
        hidden_states = outputs[2]

        for occurrence_idcs in bert_target_idcs:
            for occurrence_idx in occurrence_idcs:
                subword_representations.append(hidden_states[layer][0][occurrence_idx].cpu())

    # average representation of multiple wordpieces
    mention_representation = np.average([t.numpy() for t in subword_representations], axis=0) 

    return mention_representation



def organize_reps_and_calculate_similarity(representations_by_speaker, speaker_to_speaker_type, two_sides):
    minimum_per_side = 3
    mask_types = ['no-mask','one-mask','multi-mask']
    reps_by_side = dict()
    reps_by_side_and_by_half = dict() # for calculating the measures later
    similarity_dict = dict() # for the mask strategy evaluation

    # organize representations by mask_type, speaker_type and the half of the debate where they are found
    middle_token_num = len(coref_info['document']) // 2 # first, find the middle point of the debate
    for mask_type in mask_types:
        reps_by_side_and_by_half[mask_type] = dict()
        for speaker in representations_by_speaker:
            speaker_type = speaker_to_speaker_type[speaker]
            if speaker_type not in reps_by_side_and_by_half[mask_type]:
                reps_by_side_and_by_half[mask_type][speaker_type] = {'first-half':[], 'second-half':[]}
            for rep in representations_by_speaker[speaker][mask_type]:
                if rep['mention_start'] < middle_token_num and rep['mention_end'] < middle_token_num:
                    reps_by_side_and_by_half[mask_type][speaker_type]['first-half'].append(rep)
                elif rep['mention_start'] >= middle_token_num and rep['mention_end'] >= middle_token_num:
                    reps_by_side_and_by_half[mask_type][speaker_type]['second-half'].append(rep)


    ### Prepare the representations for the mask strategy evaluation
    # Get individual representations by side and mask_type
    for mask_type in mask_types:
        reps_by_side[mask_type] = dict()
        for speaker in representations_by_speaker:
            speaker_type = speaker_to_speaker_type[speaker]
            if speaker_type not in reps_by_side[mask_type]:
                reps_by_side[mask_type][speaker_type] = []
            reps_by_side[mask_type][speaker_type].extend(representations_by_speaker[speaker][mask_type])

    # first find the size of the side for which we have the least data. mask_type doesn't matter here
    smallest_size_found = min([len(reps_by_side['no-mask'][speaker_type]) for speaker_type in reps_by_side['no-mask']])  # for, against

    # The smallest side should have at least 6 instances. If it doesn't, we have to omit this word.
    if smallest_size_found < (minimum_per_side * 2):
        return similarity_dict, reps_by_side_and_by_half  # similarity dict is empty at this point

    #############################################################
    ######## BETWEEN 1 and 2, WITHIN-FOR, WITHIN-AGAINST ########
    #############################################################
    between_similarities_for_this_cluster = dict()
    within_similarities_for_this_cluster = dict()
    for mask_type in mask_types:
        if mask_type not in between_similarities_for_this_cluster:
            between_similarities_for_this_cluster[mask_type] = dict() # 1, 2
            within_similarities_for_this_cluster[mask_type] = dict() # for, against

    # Make random splits for each side (here mask_type is irrelevant)
    mention_idcs_for = [rep['mention_num'] for rep in reps_by_side[mask_type]['for']]
    mention_idcs_against = [rep['mention_num'] for rep in reps_by_side[mask_type]['against']]
    random.shuffle(mention_idcs_for)
    random.shuffle(mention_idcs_against)
    amount_per_side_per_comparison = smallest_size_found // 2

    # every instance set should have the same size
    idcs = {1: dict(), 2: dict()}
    idcs[1]['for'] = mention_idcs_for[:amount_per_side_per_comparison]
    idcs[2]['for'] = mention_idcs_for[amount_per_side_per_comparison:amount_per_side_per_comparison+amount_per_side_per_comparison]
    idcs[1]['against'] = mention_idcs_against[:amount_per_side_per_comparison]
    idcs[2]['against'] = mention_idcs_against[amount_per_side_per_comparison:amount_per_side_per_comparison+amount_per_side_per_comparison]

    for mask_type in mask_types:
        avg_reps = {1: dict(), 2: dict()}
        for num in [1,2]:
            for stance in idcs[num]:
                reps = [rep for rep in reps_by_side[mask_type][stance] if rep['mention_num'] in idcs[num][stance]]
                avg_reps[num][stance] = np.average([rep['representation'] for rep in reps], axis=0)

        # Now calculate the similarities
        between_similarities_for_this_cluster[mask_type][1] = {"similarity": 1 - cosine(avg_reps[1]['for'], avg_reps[1]['against']), 'idcs_by_set':[idcs[1]['for'], idcs[1]['against']]}
        between_similarities_for_this_cluster[mask_type][2] = {"similarity": 1 - cosine(avg_reps[2]['for'], avg_reps[2]['against']), 'idcs_by_set':[idcs[2]['for'], idcs[2]['against']]}
        for stance in two_sides:
            within_similarities_for_this_cluster[mask_type][stance] = {'similarity': 1 - cosine(avg_reps[1][stance], avg_reps[2][stance]), 'idcs_by_set':[idcs[1][stance], idcs[2][stance]]}

    for mask_type in mask_types:
        similarity_dict[mask_type] = {"BETWEEN": between_similarities_for_this_cluster[mask_type], "WITHIN": within_similarities_for_this_cluster[mask_type]}

    return similarity_dict, reps_by_side_and_by_half

def save_similarities(out_dir, conv_name, similarities_for_each_cluster):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with open(os.path.join(out_dir, conv_name), "w") as outfile:
        json.dump(similarities_for_each_cluster, outfile)

if __name__ == "__main__":

    files_with_problems = set()
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="debates_full_chains")
    parser.add_argument("--out_dir", default="bert_representations")
    args = parser.parse_args()

    file_extension = ".jsonlines"
    accepted_speaker_types = ["for", "against"]

    # load bert
    config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model = BertModel.from_pretrained("bert-base-uncased", config=config)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    bert_model.to(device)
    problematic_mentions = dict()

    i = -1
    for conv_name in os.listdir(args.in_dir):
        print(conv_name)
        if '8444' not in conv_name:
            continue
        i += 1
        coref_info = load_json_debate(args.in_dir, conv_name)
        similarities_for_each_cluster = []
        reps_for_each_cluster = []
        for clusternum, cluster in enumerate(coref_info["clusters"]):
            #print("cluster", clusternum)
            representations_by_speaker = dict()
            speaker_to_speaker_type = dict()
            for mention_num, (mention_start, mention_end) in enumerate(cluster['mentions']):
                # recover the original sentence and the exact idcs of the mention to extract representations
                speaker_type = cluster['speaker_types'][mention_num]
                speaker = cluster['speakers'][mention_num]
                speaker_to_speaker_type[speaker] = speaker_type
                if speaker_type not in accepted_speaker_types:
                    continue

                # extract the sentence of this mention
                found_sentence = [] # here will be token idcs of the original sentence
                for utt in coref_info['tokens_by_sentence_and_utterance']:
                    for sent in utt:
                        if mention_start in sent and mention_end in sent:
                            found_sentence = sent
                            break
                    if found_sentence:
                        break
                if not found_sentence:
                    # there were a few cases where a mention had been split across two sentences. In this case we pick only first token idx (and the fist sentence)
                    files_with_problems.add(conv_name)
                    if conv_name not in problematic_mentions:
                        problematic_mentions[conv_name] = defaultdict(list)
                    problematic_mentions[conv_name][clusternum].append((mention_start, mention_end))
                    cluster['mentions'][mention_num] = [mention_start, mention_start]
                    mention_end = mention_start
                    for utt in coref_info['tokens_by_sentence_and_utterance']:
                        for sent in utt:
                            if mention_start in sent:
                                found_sentence = sent
                                break

                original_sentence_tokens = coref_info['document'][found_sentence[0]:found_sentence[-1]+1]  # tokens of the original sentence

                # make a map to find the indices IN THE SENTENCE that correspond to the mention.
                mention_start_insentence = found_sentence.index(mention_start)
                mention_end_insentence = found_sentence.index(mention_end)
                mention_text = original_sentence_tokens[mention_start_insentence:mention_end_insentence+1]

                # bert-tokenize the sentence and get a mapping from bert-indices to insentence-idcs.
                berttok_sent, map_ori_to_bert, incomplete = smart_tokenization(original_sentence_tokens, bert_tokenizer, maxlen=bert_model.config.max_position_embeddings)
                if incomplete:
                    pdb.set_trace() # no sentence was too long not to include the mention within the first maxlen subtokens
               
                # identify the indices in the bert-tokenized sentence where our target word is
                bert_target_idcs = []
                for ori_idx in range(mention_start_insentence, mention_end_insentence+1):
                    bert_target_idcs.append(map_ori_to_bert[ori_idx])

                ### Try different masking strategies: no-mask, one-mask, multi-mask

                # multi-mask
                multimask_berttok_sent = berttok_sent[:]
                for idcs in bert_target_idcs:
                    for idx in idcs:
                        multimask_berttok_sent[idx] = bert_tokenizer.mask_token

                # one-mask
                onemask_berttok_sent = berttok_sent[:bert_target_idcs[0][0]] + [bert_tokenizer.mask_token] + berttok_sent[bert_target_idcs[-1][-1]+1:]
                onemask_bert_target_idcs = [[bert_target_idcs[0][0]]] # with one-mask, target indices change

                # Extract representation
                layer = 10
                mention_representations_by_mask_type = dict()

                mention_representations_by_mask_type['no-mask'] = extract_bert_rep(berttok_sent, bert_model, bert_tokenizer, bert_target_idcs, layer)
                mention_representations_by_mask_type['one-mask'] = extract_bert_rep(onemask_berttok_sent, bert_model, bert_tokenizer, onemask_bert_target_idcs, layer)
                mention_representations_by_mask_type['multi-mask'] = extract_bert_rep(multimask_berttok_sent, bert_model, bert_tokenizer, bert_target_idcs, layer)

                if speaker not in representations_by_speaker:
                    representations_by_speaker[speaker] = {'no-mask':[], 'one-mask':[], 'multi-mask':[]}
                for mask_type in mention_representations_by_mask_type:
                    representations_by_speaker[speaker][mask_type].append({'mention_num':mention_num, 'representation':mention_representations_by_mask_type[mask_type],
                                                                           'mention_start':mention_start, "mention_end":mention_end})



            ##### end of mention loop: going back to cluster level
            similarities_for_mask_testing, reps_by_side_and_by_half = organize_reps_and_calculate_similarity(representations_by_speaker, speaker_to_speaker_type, accepted_speaker_types)

            similarities_for_each_cluster.append({"type": cluster['type'], "cluster_name":cluster['cluster_name'], "similarity": similarities_for_mask_testing})
            reps_for_each_cluster.append(reps_by_side_and_by_half)


        save_similarities(args.out_dir, conv_name, similarities_for_each_cluster) # save similarities as jsonlines files as well
        pickle.dump(reps_for_each_cluster, open(args.out_dir + conv_name.split(".")[0] + ".pkl", 'wb')) # save representations in a pickle file

        #print("problematic mentions:", problematic_mentions)
