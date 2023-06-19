# Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques
This GitHub repository contains the official source code for Investigating Bias in Multilingual Language Models: Cross-Lingual Transfer of Debiasing Techniques.


## Required Datasets
for the different debiasing techniques, different wikipedia data is required.


|Dataset | Download | Notes | Directory|
|--------|----------|-------|----------|
|Wikipedia 2.5 en |[Download](https://drive.google.com/file/d/15Cm8E9_ZfvBhztQwjlk0GuJIjkxQSt_V/view?usp=sharing)| English Wikipedia dump used for SentenceDebias and INLP. (Meade et al., 2021) |'data/text'|
|Wikipedia 10 en  |[Download](https://drive.google.com/file/d/1NbQPD5236_LOUiHikMlVvi53gi07u2kE/view?usp=sharing)| English Wikipedia dump used for CDA and Dropout. (Meade et al., 2021) |'data/text'|
|Wikipedia 2.5 fr |[Download](https://drive.google.com/file/d/12M-ClC97-HtLHol9WDtV2-EKFOi2JFBP/view?usp=sharing)| French Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 fr  |[Download](https://drive.google.com/file/d/1sfzULqSeOfbBk_OKqto9tCB8ETl0cd__/view?usp=sharing)| French Wikipedia dump used for CDA and Dropout. |'data/text'|
|Wikipedia 2.5 de |[Download](https://drive.google.com/file/d/17Xt6rZO63pdtNA4wvZ6FKkfbUzo5hchw/view?usp=sharing)| German Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 de  |[Download](https://drive.google.com/file/d/1zzA-nxbQh2uP81hk4PI5gn9f3YcEVGuM/view?usp=sharing)| German Wikipedia dump used for CDA and Dropout. |'data/text'|
|Wikipedia 2.5 nl |[Download](https://drive.google.com/file/d/1e-4iJBEkLE53ZH9NnnNi9kSAcTiy1PBp/view?usp=sharing)| Dutch Wikipedia dump used for SentenceDebias and INLP. |'data/text'|
|Wikipedia 10 nl  |[Download](https://drive.google.com/file/d/1PbSYTpweuyHN1oDQXqQKxx7Zyr8i2Y_I/view?usp=sharing)| Dutch Wikipedia dump used for CDA and Dropout. |'data/text'|

# Acknowledgements
This code is based on the GitHub repository of Meade, N., Poole-Dayan, E., & Reddy, S. (2022, May). [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models.](https://github.com/McGill-NLP/bias-bench/tree/main). In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 1878-1898).). arXiv preprint arXiv:2110.08527. <br>
Moreover, this code contains code of Sheng Liang, Philipp Dufter, and Hinrich Schütze. 2020. [Monolingual and Multilingual Reduction of Gender Bias in Contextualized Representations.](https://github.com/liangsheng02/densray-debiasing/tree/publish) In Proceedings of the 28th International Conference on Computational Linguistics, pages 5082–5093, Barcelona, Spain (Online). International Committee on Computational Linguistics.

We thank the authors for making their code publicly available.
