

import os
import json
import pickle
import re
import pandas as pd

def load_representations(in_dir):        
    all_reps = dict()
    for fn in os.listdir(in_dir):        
        if "pkl" in fn:
            reps = pickle.load(open(os.path.join(in_dir, fn), 'rb'))
            all_reps[fn.split(".")[0]] = reps
    return all_reps


def load_similarities(in_dir):    
    all_data = dict()    
    for fn in os.listdir(in_dir):
        if "json" in fn:
            with open(os.path.join(in_dir, fn)) as f:
                d = json.load(f)                
                all_data[fn.split(".")[0]] = d
    return all_data

def load_cluster_info(in_dir):    
    cluster_data = dict()
    for fn in os.listdir(in_dir):
        if "json" in fn:
            with open(os.path.join(in_dir, fn)) as f:
                d = json.load(f)            
                cluster_data[fn.split(".")[0]] = d
    return cluster_data


def load_tfidf(in_dir, target):
    '''Load previously calculated tfidf values'''
    fn = in_dir + target + "_tfidf.tsv"
    with open(fn) as f:
        d = {line.split("\t")[0]:float(line.split("\t")[1]) for line in f.readlines()}        
    return d 




def calculate_deltas(results):
    for_delta = results['post']['for'] - results['pre']['for']
    ag_delta = results['post']['against'] - results['pre']['against']
    return for_delta, ag_delta

# a function that calculates the winning side for each debate
def obtain_winning_sides():
    winners = dict()
    all_results = dict()
    from convokit import Corpus, download
    corpus = Corpus(filename=download("iq2-corpus"))
    for conversation in corpus.iter_conversations():
        results = conversation.retrieve_meta('results')
        for_delta, ag_delta = calculate_deltas(results)
        if for_delta > ag_delta:
            winner = "for"
        elif for_delta < ag_delta:
            winner = "against"
        elif for_delta == ag_delta:
            winner = "tie"
                                                     
        winners[conversation.get_id()] = winner
        results['winner'] = winner
        results['tightness'] = abs(for_delta - ag_delta)        
        all_results[conversation.get_id()] = results
    return winners, all_results
    
    
    
# gotten from https://www.w3resource.com/python-exercises/python-functions-exercise-7.php
def num_uppercase(s):
    d={"UPPER_CASE":0, "LOWER_CASE":0}
    for c in s:
        if c.isupper():
           d["UPPER_CASE"]+=1
        elif c.islower():
           d["LOWER_CASE"]+=1
        else:
           pass
    return d["UPPER_CASE"]
    #print ("Original String : ", s)
    #print ("No. of Upper case characters : ", d["UPPER_CASE"])
    #print ("No. of Lower case Characters : ", d["LOWER_CASE"])
    
    
def load_iq2():
    '''This function loads the Intelligence Squared data and cleans some empty and corrupted utterances'''
    from convokit import Corpus, download
    corpus = Corpus(filename=download("iq2-corpus"))
    problematic_title = "smart technology is making us dumb"
    all_data = []
    for conversation in corpus.iter_conversations():
        utterance_ids = conversation.get_utterance_ids()
        for utt_id in utterance_ids:
            utterance = conversation.get_utterance(utt_id)
            if utterance.text:
                text = utterance.text
                # corrupted text tends to contain the motion (the title of the debate) in the middle of a sentence
                text = text.replace("The Rosenkranz Foundation - Intelligence Squared US Debate ?Performance Enhancing Drugs in Competitive Sports? ", "")
                text = text.replace("The Rosenkranz Foundation - Intelligence Squared US Debate “Performance Enhancing Drugs in Competitive Sports”", "")
                text = text.replace("The Rosenkranz Foundation - Intelligence Squared US Debate ? Performance Enhancing Drugs in Competitive Sports ? ", "")
                text = text.replace("The Rosenkranz Foundation - Intelligence Squared US Debate ?Performance Enhancing Drugs in Competitive Sports?", "")
                text = text.replace("Rosenkranz Foundation - Intelligence Squared U.S. - Beware the Dragon: A booming China spells trouble for America Globalization","")
                text = text.replace("Rosenkranz Foundation - Intelligence Squared U.S. - Beware the Dragon: A booming China spells trouble for America","")
                text = text.replace("The Rosenkranz Foundation - Intelligence Squared US Debate “" + conversation.meta['title'] +  "”", "")
                text = text.replace("Rosenkranz Foundation-Intelligence Square U.S.-","")
                text = text.replace("Rosenkranz Foundation-Intelligence Squared U.S.","")
                text = text.replace("Rosenkranz Foundation-Intelligence SquaredUS-Undocumented Immigrants","")
                text = text.replace("“" + conversation.meta['title'] + "”", "")
                insensitive_string = re.compile(re.escape("“" + conversation.meta['title'] + "”"), re.IGNORECASE)
                text = insensitive_string.sub("", text)

                # in some cases, the casing was different
                if problematic_title in text.lower():
                    strs_to_replace = []
                    # find the indices:
                    idcs = [m.start() for m in re.finditer(problematic_title, text.lower())]
                    for idx in idcs:
                        actual_string = text[idx:idx+len(problematic_title)]
                        if num_uppercase(actual_string) >= 2:
                            strs_to_replace.append(actual_string)
                    for actual_string in strs_to_replace:
                        text = text.replace(actual_string, "")
                text = text.strip()
                text = text.replace("  "," ")

                utt_item = {'Speaker': utterance.speaker.id, 'SpeakerType': utterance.meta['speakertype'],
                        'Utterance': text, "ConvID": conversation.id, "ConvTitle": conversation.meta['title'],
                        "RowID": utterance.id}
            all_data.append(utt_item)

    # if two utterances have the same rowid, remove one (the shortest one, if they are different)
    already_ids = set()
    clean_data = []
    for utt_item in all_data:
        if utt_item['RowID'] in already_ids:
            continue
        else:
            clean_data.append(utt_item)
            already_ids.add(utt_item['RowID'])

    all_data = pd.DataFrame(clean_data)

    return all_data


        


