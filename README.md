# Measuring Lexico-Semantic Alignment in Debates

This repository contains code for the paper:

Aina Garí Soler, Matthieu Labeau and Chloé Clavel (2023). Measuring Lexico-Semantic Alignment in Debates with Contextualized Word Representations. To appear in Proceedings of the 1st Workshop on Social Influence in Conversations (SICon), Toronto, Canada, July 14


## 1. Preparing the data (IQ2 debates corpus)

First, we need to load the debates and save them in the jsonlines format that is necessary to run the coreference solver. This step is necessary even if you do not plan to run a coreference solving model afterwards.

``python convert_debates_to_codicrac_input_format.py --out_dir [OUTPUT DIRECTORY]``

One file per debate will be saved in ``--out_dir`` (by default, ``data_for_coref/``).

## 2. Running the coreference solver

We run the coreference solver on the previous step's output using scripts from [the authors' original repository](https://github.com/samlee946/utd-codi-crac2022).

The model checkpoints used were requested directly to the authors of this model (Li et al., 2022).

Note that three different models need to be used for prediction, each one on the previous model's output, in sequence 
(ar/ar-step-1/predict.py, ar-step-2, ar-step-3). Do not hesitate to contact us for more details.

## 3. Cleaning the coreference chains and extracting common word mentions

The script ``extract_and_filter_clusters.py`` filters out some of the coreference chains (e.g. those with references to panelists and a majority of 1st and 2nd person pronouns, as explained in Section 3.3 of the paper). At the same time, it processes the debates to find the words that are common to both sides.
This is to be run either on the coreference solver's output or on the output at step 1 of this readme.

``python extract_and_filter_clusters.py --in_dir [DIRECTORY WITH JSONLINES FILES] --out_dir [OUTPUT DIRECTORY]``

By default, ``--in_dir`` is ``coref_predictions`` and ``--out_dir`` is ``debates_full_chains``.
You can also modify the minimum number of utterances per speaker required to keep a word/coreference chain (``--min_utts_per_speaker``)


## 4. Calculating tf-idf

Tf-idf values can be calculated after running the script in step 3:

``python --in_dir [DIRECTORY WITH CLEAN CLUSTERS] --out_dir [OUTPUT DIRECTORY]``

By default, ``--in_dir`` is ``debates_full_chains`` and ``--out_dir`` is ``tfidf_data``.


## 5. Calculating Dialign measures

Dialign measures can also be calculated on step 3's output. First, we need to adapt the data's format:

``python convert_debates_to_dialign_format.py --in_dir [...] --out_dir [...]''

By default, ``--in_dir`` is ``debates_full_chains`` and ``--out_dir`` is ``data_for_dialign``.

You can download the dialign software from [this repository](https://github.com/GuillaumeDD/dialign).
To calculate the measures, you need to run the command:

``java -jar [PATH TO DIALIGN]/dialign.jar -i [INPUT DIRECTORY] -o [OUTPUT DIRECTORY]``

By default, ``-i`` should be ``data_for_dialign`` and ``-o`` should be ``dialign-output``.

## 6. Extracting BERT representations

After step 3, BERT representations can be extracted as follows:

``python extract_bert_representations.py --in_dir [...] --out_dir [...]``

By default, ``--in_dir`` is ``debates_full_chains`` and ``out_dir`` is ``bert_representations``.
This script creates two files for every debate:
* One jsonfile containing the cosine similarities that are used for the side evaluation described in Appendix A, 
* One pkl file with BERT representations organized by side and half.

## 7. Running the side-evaluation

To run the side evaluation described in Appendix A, where we test different masking strategies, you can run the code in the ``choosing_masking_strategy.ipynb`` Jupyter notebook.


## 8. Calculating alignment measures

In the ``calculate_alignment.ipynb`` notebook we calculate all alignment measures, calculate their descriptive statistics as well as their intercorrelations and their correlation with frequency. We also look at the results for the running example in the paper.
Measures are saved separately for each word in a pickle file ``measures_all_debates_by_word.pkl`` and at the debate-level using different types of vocabulary in ``measures_all_debates.pkl``.

## 9. Classification

The leave-one-out classification experiments can be run in the ``classification.ipynb`` notebook after obtaining the measures at step 8.


## Citation

If you use the code in this repository, please cite our paper! (BibTeX coming soon)


## Contact

For any questions or requests feel free to contact me: aina dot garisoler at telecom-paris dot fr


## References

Shengjie Li, Hideo Kobayashi, and Vincent Ng. 2022. [Neural anaphora resolution in dialogue revisited](https://aclanthology.org/2022.codi-crac.4/). In Proceedings of the CODI-CRAC 2022 Shared Task on Anaphora, Bridging, and Discourse Deixis in Dialogue, pages 32–47, Gyeongju, Republic of Korea. Association for Computational Linguistics.






