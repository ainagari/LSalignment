'''
This script reads in the debates (formatted for the coreference solver) and it calculates tfidf.
It saves one file per debate in out_dir.
'''


import os
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import argparse

def calculate_tfidf(data, targets):

    vectorizer = TfidfVectorizer(max_df=len(data)-1, token_pattern=r'\b\w\w+\b|(?<!\w)@\w+|(?<!\w)#\w+', use_idf=True)

    by_document = []
    for debate in targets:
        debate = data[debate]
        text = " ".join(["_".join(lemmapos) for lemmapos in debate['document_lemmas_and_tags']])
        by_document.append(text)

    tfidf = vectorizer.fit_transform(by_document).toarray()

    return vectorizer, tfidf


def save_tfidf_data(vectorizer, targets, target, path):
    idx_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    targetnum = targets.index(target)
    word_tfidf = dict()

    for idx, value in enumerate(tfidf[targetnum]):
        word = idx_to_word[idx]
        word_tfidf[word] = value

    # sort it from highest to lowest tfidf
    sorted_word_tfidf = sorted(word_tfidf.items(), key=lambda k:k[1], reverse=True)
    # only keep non-zero terms
    sorted_word_tfidf_nonzero = [(k,v) for k,v in sorted_word_tfidf if v > 0]

    out_path = os.path.join(path, target + "_tfidf.tsv")

    with open(out_path, 'w') as out:
        for w, t in sorted_word_tfidf_nonzero:
            uppercased_tag = "".join(w.split("_")[:-1]) + "_" + w.split("_")[-1].upper()
            out.write(uppercased_tag + "\t" + str(t) + "\n")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="debates_full_chains", help="indicate the directory with the jsonlines debate files, once lemmatized and postagged "
                                                                         "(after using the script extract_and_filter_clusters.py")
    parser.add_argument("--out_dir", default="tfidf_data")
    args = parser.parse_args()

    all_data = dict()

    for debate_filename in os.listdir(args.in_dir):
        with open(os.path.join(args.in_dir, debate_filename)) as f:
	        all_data[debate_filename.split(".")[0]] = json.load(f)


    targets = list(all_data.keys())

    vectorizer, tfidf = calculate_tfidf(all_data, targets)
    # Save
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for target in targets:
        save_tfidf_data(vectorizer, targets, target, path=args.out_dir)
