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

## Use
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

## Example
Here follows an example of how to calculate the bias direction for SentenceDebias in French using mBERT.
```
$ python experiments/sentence_debias_subspace.py \
$                 --persistent_dir=[path] \
$                 --model="BertModel" \
$                 --model_name_or_path='bert-base-multilingual-uncased'  \
$                 --bias_type="gender" \
$                 --lang_debias='fr' \
```
Once you have the bias direction, you can calculate the bias metrics for the different languages as follows. Here is an example of calculating it for the English subsample 0: 

```
$ python experiments/crows_debias.py \
$                 --persistent_dir='[path]' \
$                 --model "SentenceDebiasBertForMaskedLM" \
$                 --model_name_or_path 'bert-base-multilingual-uncased' \
$                 --bias_direction '[path]/results/subspace/subspace_m-BertModel_c-bert-base-multilingual-uncased_t-gender_debias-fr.pt' \
$                 --bias_type "gender"  \
$                 --sample="false" \
$                 --seed=0 \
$                 --lang_eval='en' \
$                 --lang_debias='fr' \
```




## Acknowledgements
This code is based on the GitHub repository of Meade, N., Poole-Dayan, E., & Reddy, S. (2022, May). [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models.](https://github.com/McGill-NLP/bias-bench/tree/main). In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1878-1898).). arXiv preprint arXiv:2110.08527. <br>
Moreover, this code contains code of Sheng Liang, Philipp Dufter, and Hinrich Schütze. 2020. [Monolingual and Multilingual Reduction of Gender Bias in Contextualized Representations.](https://github.com/liangsheng02/densray-debiasing/tree/publish) In Proceedings of the 28th International Conference on Computational Linguistics, pages 5082–5093, Barcelona, Spain (Online). International Committee on Computational Linguistics.

We thank the authors for making their code publicly available.
