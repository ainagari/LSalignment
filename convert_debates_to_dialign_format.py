'''
With this script we prepare the debates' format so it is compatible with the dialign software.
The input are the .json debate files prepared for coreference solving 
The output is saved in out_dir (data_for_dialign/).
To run dialign on the output, download the software from this repository: https://github.com/GuillaumeDD/dialign
and run the command:
java -jar [PATH TO DIALIGN]/dialign.jar -i data_for_dialign/ -o [OUTPUT DIR]
'''


import os
import json
import argparse


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="debates_full_chains")
    parser.add_argument("--out_dir", default="data_for_dialign")
    args = parser.parse_args()
    
    if not os.path.exists(args.out_dir):
    	os.mkdir(args.out_dir)

    for debatefn in os.listdir(args.in_dir):
        debate_data = json.load(open(os.path.join(args.in_dir, debatefn)))
        file_lines = []
        for utt, speakertype in zip(debate_data["tokens_by_sentence_and_utterance"], debate_data['speakertypes_by_utterance']):
            utt_tokens = []
            for sent in utt:
                utt_tokens.extend([debate_data['document'][t] for t in sent])
            file_lines.append((speakertype, utt_tokens))
            
        

        with open(os.path.join(args.out_dir, debatefn.split(".")[0] + ".tsv"), 'w') as out:
            for st, utt_tokens in file_lines:
                if st in ["for","against"]:
                    out.write(st + ":\t" + " ".join(utt_tokens) + "\n")


