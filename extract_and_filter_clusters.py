#/tsi/clusterhome/agari/anaconda3/bin/python
# -*-coding:Latin-1 -*


import argparse
import os
from convert_debates_to_codicrac_input_format import load_iq2
from collections import Counter
import json
import pdb
from nltk.corpus import stopwords
import copy
from collections import defaultdict
from operator import itemgetter
from string import punctuation

sws = set(stopwords.words('english'))
from nltk import pos_tag, WordNetLemmatizer




def assign_ref_to_speaker(coref_prediction, tokens, instances):
    new_clusters = []
    for i, cluster in enumerate(coref_prediction['clusters']):
        speakers_of_cluster = []
        speakertypes_of_cluster = []
        for refstart, refend in cluster:
            for utt_num, utt_tokens in enumerate(tokens):
                if refstart in utt_tokens:
                    # get the speaker corresponding to this utt_num
                    speaker = instances.iloc[utt_num]['Speaker']
                    speakertype = instances.iloc[utt_num]['SpeakerType']
                    speakers_of_cluster.append(speaker)
                    speakertypes_of_cluster.append(speakertype)
        new_cl = {'type': 'coref', 'speakers': speakers_of_cluster, 'mentions': cluster,
                  'speaker_types': speakertypes_of_cluster}
        new_clusters.append(new_cl)

    coref_prediction['clusters'] = new_clusters

    return coref_prediction


def find_and_remove_automatic_pronoun_clusters(coref_prediction):
    limit = 0.7
    ## Determining if a cluster contains a majority (>= limit) of 1st and 2nd pronouns
    cluster_types = []
    I_pronouns = ["i", 'me', "my", "mine"]
    you_pronouns = ["you", "your", "yours"]
    for i, cluster in enumerate(coref_prediction['clusters']):
        amount_I_pronouns = 0
        amount_you_pronouns = 0
        actual_words = []
        for refstart, refend in cluster['mentions']:
            actual_word = coref_prediction['document'][refstart:refend + 1]  # actual_word is a list
            actual_words.append([w.lower() for w in actual_word])

        # count 1st and 2nd person pronouns in cluster
        for ws in actual_words:
            for pron in I_pronouns:
                if pron in ws:
                    amount_I_pronouns += 1
                    break
            for pron in you_pronouns:
                if pron in ws:
                    amount_you_pronouns += 1
                    break

        if amount_I_pronouns / len(actual_words) >= limit:
            cluster_type = 'I'
        elif amount_you_pronouns / len(actual_words) >= limit:
            cluster_type = 'you'
        elif (amount_I_pronouns + amount_you_pronouns) >= limit:
            cluster_type = 'pron'
        else:
            cluster_type = cluster['type']

        cluster_types.append(cluster_type)

    # No distinction between I-you-pron: we remove them all
    for cl, cltype in zip(coref_prediction['clusters'], cluster_types):
        cl['type'] = cltype
    new_clusters = []
    for cl in coref_prediction['clusters']:
        if cl['type'] not in ["I", "you", "pron"]:
            new_clusters.append(cl)
    coref_prediction['clusters'] = new_clusters

    return coref_prediction


def save_found_clusters_one_conv(prediction, out_dir, conv_name):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, conv_name + ".jsonlines"), 'w') as f:
        json.dump(prediction, f, ensure_ascii=False)


def clean_panelist_clusters(instances, pred):
    i_pronouns = {'i', 'my', 'mine', 'myself'}
    pronouns = {'he','him','she','her','himself','herself','you','your','yours','yourself','i','my','mine','myself'}
    panelists = set(instances.Speaker.unique())
    if "audience" in panelists:
        panelists.remove("audience")
    speaker_to_type = dict(list(zip(instances.Speaker, instances.SpeakerType)))
    new_clusters = []
    ### Loop over clusters: detect if cluster refers to panelist
    for clnum, cl in enumerate(pred['clusters']):
        speakers_found = set()
        cluster_mentions = []
        for refstart, refend in cl['mentions']:
            mention_text = " ".join(pred['document'][refstart:refend+1])
            cluster_mentions.append(mention_text)
            if mention_text in panelists:
                speakers_found.add(mention_text)

        if not speakers_found: # If the cluster does not correspond to a panelist, we keep it
            new_cluster = cl
            new_clusters.append(new_cluster)
        elif speakers_found: # The cluster refers to some panelist - omit it
            continue

    pred['clusters'] = new_clusters

    return pred

def contains_punctuation(word):
    for char in word:
        if char in punctuation:
            return True
    return False



def word_frequencies_by_speaker(tokens, tokennums_by_utterance, all_speakers, all_speaker_types, min_utts_by_speaker):
    '''
    Count word usage by each speakertype (for vs against), select words that are used min_utts_by_speaker times by each side and which are not stopwords
    keep only nouns, verbs, remove stopwords
    The cluster name will be the lemma
    '''
    document_lemmas_and_tags = []
    mentions_by_speakertype = dict()
    lemmatizer = WordNetLemmatizer()
    for utt, st, sp in zip(tokennums_by_utterance, all_speaker_types, all_speakers):
        if not utt:
            continue
        if st not in mentions_by_speakertype:
            mentions_by_speakertype[st] = dict()
        if sp not in mentions_by_speakertype[st]:
            mentions_by_speakertype[st][sp] = defaultdict(list)

        utt_words = tokens[utt[0]:utt[-1] + 1]

        postagged_utt_words = pos_tag(utt_words, tagset='universal')

        # keep only nouns and verbs
        utt_lemmas = []
        for (w, t), token_idx in zip(postagged_utt_words, utt):
            if t[0].lower() in  set(["a", "s", "r", "n", "v"]):
                lemma = lemmatizer.lemmatize(w.lower(), t[0].lower())
            else:
                lemma = w.lower()
            document_lemmas_and_tags.append((lemma,t))
            if t in ["NOUN", "VERB"] and w not in sws and w not in punctuation and not contains_punctuation(w):
                lemma_pos = lemma + "_" + t
                mentions_by_speakertype[st][sp][lemma_pos].append([token_idx, token_idx])

    exclude_from_min_counts = ["mod","host","unknown"]

    potential_keys = []
    freqs = dict()
    for st in mentions_by_speakertype:
        if st in exclude_from_min_counts:
            continue
        for sp in mentions_by_speakertype[st]:
            for w in mentions_by_speakertype[st][sp]:
                if w not in freqs:
                    freqs[w] = defaultdict(int)
                freqs[w][st] += len(mentions_by_speakertype[st][sp][w])

    common_words = [w for w in freqs if min(freqs[w].values()) >= min_utts_by_speaker]

    #finally, create the new clusters
    new_clusters = []
    for w in common_words:
        new_cluster = {'mentions':[], 'speakers':[], 'speaker_types':[], "type":"word", 'cluster_name': w} # w is lemma_pos
        for st in mentions_by_speakertype:
            for sp in mentions_by_speakertype[st]:
                if w in mentions_by_speakertype[st][sp]:
                    mentions_this_speaker = mentions_by_speakertype[st][sp][w]
                    new_cluster['mentions'].extend(mentions_this_speaker)
                    new_cluster['speakers'].extend([sp]*len(mentions_this_speaker))
                    new_cluster['speaker_types'].extend([st] * len(mentions_this_speaker))
        new_clusters.append(new_cluster)

    return new_clusters, document_lemmas_and_tags


def get_conv_data(data, convname):
    conv_instances = data[data["ConvID"] == convname]
    conv_utterances = conv_instances["Utterance"]
    conv_text = " ".join(conv_utterances)

    return conv_text, conv_utterances, conv_instances


def remove_clusters_with_insufficient_instances(coref_pred, min_utts_by_speaker):
    irrelevant_speaker_types = ['mod', 'host', 'unknown']
    kept_clusters = []
    for i, cl in enumerate(coref_pred['clusters']):
        c = Counter(cl['speaker_types'])
        for ist in irrelevant_speaker_types:
            del c[ist]
        if c:
            minval = min(c.values())
            if len(c) < 2 or minval < min_utts_by_speaker:  # if there's only one speaker or there are less than 3 instances for one speaker: omit it
                continue
            else:
                kept_clusters.append(i)

    #print("number of clusters that involve two speakers, with at least", min_utts_by_speaker, " refs each:", len(kept_clusters))
    final_pred = copy.copy(coref_pred)
    final_pred['clusters'] = []
    for clnum, cl in enumerate(coref_pred['clusters']):
        if clnum in kept_clusters:
            final_pred['clusters'].append(cl)
    return final_pred


def remove_repeated_instances(pred):
    new_clusters = []
    clusters_with_duplicates = 0
    for clnum, cl in enumerate(pred['clusters']):
        # get pairs of (mention, sentence) in the cluster. Detect duplicates, remove them.
        men_sens = []
        for midx, mention_nums in enumerate(cl['mentions']):
            mention_words = pred["document"][mention_nums[0]:mention_nums[1] + 1]
            mention_sentence_nums = [s for utt in pred['tokens_by_sentence_and_utterance'] for s in utt if mention_nums[0] in s and mention_nums[1] in s]
            if not mention_sentence_nums:
                continue
            else:
                mention_sentence_nums = mention_sentence_nums[0]
                mention_sentence_words = [pred["document"][i] for i in mention_sentence_nums]

            men_sens.append((tuple(mention_words), tuple(mention_sentence_words), tuple(mention_sentence_nums), cl['speakers'][midx]))

        # Now detect duplicates and remove them.
        # it is not a duplicate if it is the same word happening TWICE in the SAME sentence.
        found_duplicates = []
        for i, (mw, msw, msn, sp) in enumerate(men_sens):
            for j, (other_mw, other_msw, other_msn, other_sp) in enumerate(men_sens):
                if i >= j:
                    continue
                if mw == other_mw and msw == other_msw and msn != other_msn and sp == other_sp:
                    found_duplicates.append((i, j))
        if not found_duplicates:
            new_clusters.append(cl)
        else: # remove duplicates carefully
            #pdb.set_trace()
            clusters_with_duplicates += 1
            new_cluster = {'mentions': [], 'speakers': [], 'speaker_types': [], 'type': cl['type']}
            found_nplicates = []
            for k, (m1, m2) in enumerate(found_duplicates):
                for_m1 = [m1, m2]
                for l, (other_pair) in enumerate(found_duplicates):
                    if l <= k:
                        continue
                    if m1 in other_pair and m2 not in other_pair:
                        other_m = [m for m in other_pair if m != m1][0]
                        for_m1.append(other_m)
                found_nplicates.append(sorted(for_m1))

            to_remove = []
            for ms in found_nplicates:
                to_remove.extend(ms[1:])
            pdb.set_trace()

            if not men_sens:
                pdb.set_trace()
            for i, mensen in enumerate(men_sens):
                if i in to_remove:
                    continue
                else:
                    new_cluster['mentions'].append(cl['mentions'][i])
                    new_cluster['speakers'].append(cl['speakers'][i])
                    new_cluster['speaker_types'].append(cl['speaker_types'][i])

            if new_cluster['mentions'] == []:
                pdb.set_trace()
            new_clusters.append(new_cluster)

    pred['clusters'] = new_clusters
    #print("there were", clusters_with_duplicates, "clusters with duplicates") 

    return pred



def assign_cluster_names(coref_pred):
    '''Tjhis function assigns names to clusters coming from the coreference solver
    # we pick at most three mentions: the first mention, the longest one, and the most frequent one.
    '''
    for cl in coref_pred['clusters']:
        if cl['type'] != "word":
            words_defining_cluster = []
            for mention_start, mention_end in cl['mentions']:
                words_defining_cluster.append(coref_pred['document'][mention_start:mention_end+1])

            # pick the first mention for this cluster
            best_words_defining_cluster = [tuple(words_defining_cluster[0])]
            # sort mentions by length
            sorted_words_defining_cluster_bylength = [tuple(w[0]) for w in sorted([(w, len(w)) for w in words_defining_cluster[1:]],
                                                                key=itemgetter(1), reverse=True)]
            # sort mentions by frequency
            c = Counter([tuple(w) for w in words_defining_cluster])
            sorted_words_defining_cluster_byfreq = [tuple(w[0]) for w in sorted([(w, c[tuple(w)]) for w in words_defining_cluster[1:]],
                                                              key=itemgetter(1), reverse=True)]

            # now put the three together (if they are the same one we will have fewer than 3)
            best_words_defining_cluster.append(sorted_words_defining_cluster_bylength[0])
            best_words_defining_cluster.append(sorted_words_defining_cluster_byfreq[0])
            best_words_defining_cluster = list(dict.fromkeys(best_words_defining_cluster))  # to keep an ordered list of unique mentions
            best_words_defining_cluster = " // ".join([" ".join(w) for w in best_words_defining_cluster])

            cl['cluster_name'] = best_words_defining_cluster

    return coref_pred



def preserve_chronology(coref_pred):
    '''
    This function makes sure that all clusters have their mentions in chronological order, for convenience.
    '''
    new_clusters = []
    for cl in coref_pred['clusters']:
        starts = [mention_start for mention_start, me in cl['mentions']]
        if starts == sorted(starts): # if chronological order is respected: keep the cluster as is
            new_clusters.append(cl)
        else: # otherwise: fix it
            new_cluster = copy.copy(cl)
            new_cluster['mentions'], new_cluster['speakers'], new_cluster['speaker_types'] = (list(l) for l in zip(*sorted(zip(cl['mentions'], cl['speakers'], cl['speaker_types']))))
            new_clusters.append(new_cluster)

    coref_pred['clusters'] = new_clusters
    return coref_pred


def load_prediction(diri, convname):
    #data = []
    if not convname + ".jsonlines" in os.listdir(diri):
        return None
    with open(os.path.join(diri, convname + ".jsonlines")) as f:
        #for l in f:
        data = json.load(f)
    #assert len(data) == 1
    return data #[0]


def fix_codicrac_mention_indices(pred):
    '''If we ran the coreference solver, which works on subtokens, get the indices corresponding to the full tokens'''
    all_doc = []
    for s in pred['sentences']:
        all_doc.extend(s)

    sentence_limits = []
    for s in pred['sentences']:
        if sentence_limits:
            sentence_limits.append(sentence_limits[-1] + len(s))
        else:
            sentence_limits.append(len(s))
    new_clusters = []

    for cluster in pred['predicted_clusters']:
        new_cluster = []
        for mention_start, mention_end in cluster:
            sentence_idx = [i for i, limit in enumerate(sentence_limits) if mention_start <= limit and mention_end <= limit]
            sentence_idx = sentence_idx[0]
            to_add = (sentence_idx * 2) + 1
            actual_mention_start = pred['subtoken_map'][mention_start - to_add]
            actual_mention_end = pred['subtoken_map'][mention_end - to_add]
            new_cluster.append([actual_mention_start, actual_mention_end])
        new_clusters.append(new_cluster)

    return new_clusters





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="coref_predictions", help="indicate the directory with the jsonlines debate files, with or without the coreference chains")
    parser.add_argument("--out_dir", default="debates_full_chains", help="minimum number of utterances per speaker")
    parser.add_argument("--min_utts_per_speaker", default=3, help="minimum number of utterances per speaker")
    args = parser.parse_args()


    # Loading data
    data = load_iq2()

    # Loop over debates  
    for convname in data["ConvID"].unique():
        print(convname)
        pred = load_prediction(args.in_dir, convname)
        if not pred:
            print("File not available", convname)
            continue

        # If the coref solver was run, this maps indices to tokens (instead of subtokens)
        if 'predicted_clusters' in pred:
            pred['clusters'] = fix_codicrac_mention_indices(pred)
            del pred['predicted_clusters']
        pred['document'] = copy.copy(pred['tokens'])
        del pred['tokens']

        ### Prepare conversation data
        conv_texts, conv_utterances, conv_instances = get_conv_data(data, convname)

        tokens_by_utterance = []
        for utt in pred['tokens_by_sentence_and_utterance']:
            flat_utt = []
            for sent in utt:
                flat_utt.extend(sent)
            tokens_by_utterance.append(flat_utt)

        all_speakers = conv_instances['Speaker']
        all_speaker_types = conv_instances['SpeakerType']

        pred['speakers_by_utterance'] = list(all_speakers)
        pred['speakertypes_by_utterance'] = list(all_speaker_types)

        # Obtain speaker information for every coref cluster (who said what?). Complete predictions with this info
        pred = assign_ref_to_speaker(pred, tokens_by_utterance, conv_instances)  # the output is a dict

        # Remove repeated instances, if there are any left
        pred = remove_repeated_instances(pred)

        # Exclude clusters referring to panelists
        pred = clean_panelist_clusters(conv_instances, pred)

        ### Find the common words (V_W) and create clusters for them
        word_clusters, document_lemmas_and_tags = word_frequencies_by_speaker(pred['document'], tokens_by_utterance, all_speakers, all_speaker_types, args.min_utts_per_speaker)
        pred['clusters'].extend(word_clusters)
        pred['document_lemmas_and_tags'] = document_lemmas_and_tags


        # remove clusters containing mostly pronouns
        clean_pred = find_and_remove_automatic_pronoun_clusters(pred)

        ### Only keep clusters that have enough speakers
        # We do this by looking at SpeakerType. We want to ignore moderators, hosts and 'unknown' (mod, host, unknown).
        # For the rest (for/against) we want each to have at least min_utts_by-speaker utterances in a cluster. Otherwise, we omit it.
        final_pred = remove_clusters_with_insufficient_instances(clean_pred, args.min_utts_per_speaker)

        final_pred = assign_cluster_names(final_pred)
        final_pred = preserve_chronology(final_pred)

        convname_for_file = str(convname)

        # Save
        save_found_clusters_one_conv(final_pred, args.out_dir, convname_for_file)


