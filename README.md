# Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques
This GitHub repository contains the official source code for Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques.

## Installation
```
$ conda create --name CrossLingualBias python=3.7
$ conda activate CrossLingualBias
$ pip install -r requirements.txt
```

## Required Datasets
For the different debiasing techniques, different wikipedia data is required. These should be put in the appropriate directory.


|Dataset | Download | Notes | Directory|
|--------|----------|-------|----------|
|Wikipedia 2.5 en |[Download](https://drive.google.com/file/d/1nGcRFOBep_M7HjvC_qM-9JFee_rWQRQO/view?usp=sharing)| English Wikipedia dump used for SentenceDebias and INLP. (Meade et al., 2021) |'data/text'|
|Wikipedia 10 en  |[Download](https://drive.google.com/file/d/1yQbZMGuUa3taP_xoGThRq0vkb9Kj0uC-/view?usp=sharing)| English Wikipedia dump used for CDA and Dropout. (Meade et al., 2021) |'data/text'|
|Wikipedia 2.5 fr |[Download](https://drive.google.com/file/d/1TAQYkB9kniSX5-2IppPJR8xiTbMFRwrx/view?usp=sharing)| French Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 fr  |[Download](https://drive.google.com/file/d/1HEQ-55kH4BIGBHU_84FsyMZwLg3kgwJX/view?usp=sharing)| French Wikipedia dump used for CDA and Dropout. |'data/text'|
|Wikipedia 2.5 de |[Download](https://drive.google.com/file/d/1RRizrCShzT7yk8hRMDN6Zj-HoyfqQkPt/view?usp=sharing)| German Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 de  |[Download](https://drive.google.com/file/d/1pvKXfK-oyfE-_j1M3BL4LD94XT10p4go/view?usp=sharing)| German Wikipedia dump used for CDA and Dropout. |'data/text'|
|Wikipedia 2.5 nl |[Download](https://drive.google.com/file/d/1jCUWl0kT0TJsljeMZvZEkC4tEWjSxMM8/view?usp=sharing)| Dutch Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 nl  |[Download](https://drive.google.com/file/d/1Mhn0kG2MZi36CNImBNDhiiNSXh-h9-Uc/view?usp=sharing)| Dutch Wikipedia dump used for CDA and Dropout. |'data/text'|

## Experiments
The different experiments can be found in the 'experiments' directory. Using the following three files, the bias directions/projection matrices for densray, sentencedebias, and inlp can be calculated.
* densray_subspace.py
* sentencedebias_subspace.py 
* inlp_projection_matrix.py

Using this file, the additional pretraining step is executed for CDA/ dropout regularization.
* run_mlm.py

To evaluate the models, the following files can be used
* crows.py
* crows_debias.py
* crows_dropout_cda.py

For all experiments, the following vocabulary seeds are used: 0, 1, 2

## Example No Debiasing
The results of mBERT on the different datasets can be calculated as follows. To get full results, all bias types and all seeds of the datasets should be run. 

```
$ python experiments/crows.py \
$                 --persistent_dir="[path]" \
$                 --model="BertForMaskedLM" \
$                 --model_name_or_path='bert-base-multilingual-uncased' \
$                 --bias_type="race" \
$                 --sample="false" \
$                 --seed=0 \
$                 --lang='fr' \
```

## Example SentenceDebias & DensRay
Here follows an example of how to calculate the bias direction for SentenceDebias in French using mBERT. For DensRay, 'sentence_debias_subspace.py' should be changed by 'densray_subspace.py'.
```
$ python experiments/sentence_debias_subspace.py \
$                 --persistent_dir=[path] \
$                 --model="BertModel" \
$                 --model_name_or_path='bert-base-multilingual-uncased'  \
$                 --bias_type="gender" \
$                 --lang_debias='fr' \
```
Once you have the bias direction, you can calculate the bias metrics for the different languages as follows. For DensRay, 'SentenceDebiasBertForMaskedLM' should be changed by 'DensrayDebiasBertForMaskedLM'

```
$ python experiments/crows_debias.py \
$                 --persistent_dir='[path]' \
$                 --model "SentenceDebiasBertForMaskedLM" \
$                 --model_name_or_path 'bert-base-multilingual-uncased' \
$                 --bias_direction '[path_to_bias_direction]' \
$                 --bias_type "gender"  \
$                 --sample="false" \
$                 --seed=0 \
$                 --lang_eval='en' \
$                 --lang_debias='fr' \
```

## Example INLP
For INLP, first a projection matrix should be calculated:
```
$python experiments/inlp_projection_matrix.py \
$                 --persistent_dir='[path]' \
$                 --model="BertModel" \
$                 --model_name_or_path='bert-base-multilingual-uncased' \
$                 --bias_type="religion" \
$                 --n_classifiers='80' \
$                 --seed='0' \
$                 --lang_debias="fr" \
```
subsequently, you can calculate the bias metrics as follows:
```
$ python experiments/crows_debias.py \
$                 --persistent_dir='[path]' \
$                 --model="INLPBertForMaskedLM" \
$                 --model_name_or_path='bert-base-multilingual-uncased' \
$                 --projection_matrix='[path_to_projection_matrix]' \
$                 --bias_type="gender"  \
$                 --sample="false" \
$                 --seed=0 \
$                 --lang_eval='en' \
$                 --lang_debias='fr' \
```

## Example CDA & Dropout

For CDA, first, an additional pretraining step should be executed in a language of your choice, for example French:
```
$ python  experiments/run_mlm.py \
$                    --model_name_or_path "bert-base-multilingual-uncased" \
$                    --cache_dir "[path]/cache/" \
$                    --do_train \
$                    --train_file "data/text/wiki-fr_sample_10.txt" \
$                    --validation_split_percentage 0 \
$                    --max_steps 2000 \
$                    --per_device_train_batch_size 4 \
$                    --gradient_accumulation_steps 128 \
$                    --max_seq_length 512 \
$                    --save_steps 500 \
$                    --preprocessing_num_workers 4 \
$                    --counterfactual_augmentation "race" \
$                    --persistent_dir "[path]" \
$                    --seed  0 \
$                    --output_dir "[path_to_dir]"

```
For dropout regularization, '--counterfactual_augmentation "race" \' should be changed by ' --dropout_debias \'
Once you have your trained model, you use:
```
$python experiments/crows_dropout_cda.py \
$                 --persistent_dir="[path]" \
$                 --model="dropout_mbert" \
$                 --model_name_or_path='[path_to_dir]' \
$                 --bias_type="gender" \
$                 --sample='false' \
$                 --seed=0 \
$                 --lang_eval='en' \
$                 --lang_debias='fr' \
$                 --seed_model=0 \
```

## Acknowledgements
This code is based on the GitHub repository of Meade, N., Poole-Dayan, E., & Reddy, S. (2022, May). [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models.](https://github.com/McGill-NLP/bias-bench/tree/main). In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1878-1898).). arXiv preprint arXiv:2110.08527. <br>
Moreover, this code contains code of Sheng Liang, Philipp Dufter, and Hinrich Schütze. 2020. [Monolingual and Multilingual Reduction of Gender Bias in Contextualized Representations.](https://github.com/liangsheng02/densray-debiasing/tree/publish) In Proceedings of the 28th International Conference on Computational Linguistics, pages 5082–5093, Barcelona, Spain (Online). International Committee on Computational Linguistics.

We thank the authors for making their code publicly available.
