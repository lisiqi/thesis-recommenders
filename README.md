# Introduction
This repository includes the main code part for my master thesis "*Implicit Feedback Based Context-Aware Recommender For Music Light System*". 

https://pure.tue.nl/ws/files/90854618/master_thesis_final_revised_SiqiLi.pdf

In music light system, users can enjoy personalized light effect conducted by smart light while listening to the music. We can model this problem as a recommendation system since essentially the light effect is the item to recommend. However, the recommender for music light system is not traditional. Compared to traditional recommender system, it has two other main characteristics: 

1. There is no explicit ratings, but we can infer user preference from the historical usage data as implicit feedback.
2. We must consider necessary music features (e.g. tempo, key) as context information.

Therefore, the music light system can be modeled as an implicit feedback based context-aware recommender system.

Currently the research of music light system is in the early stage. Since the system does not exist yet, there is no transaction data and the recommender cannot be built at this moment. Thus, the objective of this thesis is to explore potential recommendation methods that might work when the usage log is largely collected. The main research question and sub-questions are listed below:

- **MRQ** Would the context-aware recommender system methods be potentially applicable in the music light system and thus the personalized item (light effect) recommendation task can be solved?
- **RQ1** Which methods can deal with arbitrary and large number of context dimensions, and how do they model the context information?
- **RQ2** How does the performance of the recommender system change when introducing the contextual information?
- **RQ3** How do the different methods perform under cold-start situation?
- **RQ4** How does the performance of different methods get influenced by the noise of unreliable context?

The thesis is aimed at and organized by answering above questions.

# DataSet
The dataset used is [Frappe](http://baltrunas.info/research-menu/frappe). It shares the same key characteristics with future HueMusic dataset – containing both implicit feedback and rich context information. It is a large real-world dataset and appears to be the only public available dataset that contains both two characteristics so far. And it is used in the most recent researches on context-aware recommendation field.

File [frappe.ipynb](/frappe.ipynb) contains the code to examine and pre-process the dataset.

# Evaluation Metrics
Since implicit feedback based context-aware recommendation is a top-N recommendation task, I basically use four list based metrics: precision@N, MAP@N, NDCG@N and ERR@N, where N = 10 considering the size of the Frappe dataset. Precision and MAP are two classification accuracy metrics and NDCG and ERR are two ranking accuracy metrics; see Section 2.5.2 of my thesis. Note that since precision is the mean precision of recommendations for all user-context configuration, I refer to MP. Scores by all of these four metrics are the higher the better.

 - MP: Mean Precision
 - MAP: Mean Average Precision
 - NDCG: Normalized Discounted Cumulative Gain
 - ERR: Expected Reciprocal Rank

Check Section 2.5 and 4.3.2 of my thesis for detailed descussion and comparison of above four metrics and other evaluation metrics for recommender systems. 

File [measure.py](/measure.py) is the implementation of above four metrics.

**From chapter 8 of Andrew NG's new book [*Machine Learning Yearning*](http://www.mlyearning.org), one single-number evaluation metric is more preferred in practice. If we care multiple metrics, we can simply take the mean of them (not necessarily the arithmetic mean). 

# Methods
Four start-of-the-art methods are studied with experiments. The focus is on the two context-aware recommender methods: FM and GPPW.

## Method 1: iALS
The matrix factorization (MF) method specially for handling data with implicit feedback.

> Yifan Hu, Yehuda Koren, and Chris Volinsky. Collaborative filtering for implicit feedback datasets. In Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on, pages 263–272. Ieee, 2008.

In my experiments I make use of [Ben Frederickson’s Python implementation](https://github.com/benfred/implicit).
  
## Method 2: LightFM
A hybrid MF model representing users and items as linear combinations of their context features’ latent factors. 

> Maciej Kula. Metadata embeddings for user and item cold-start recommendations. In Toine Bogers and Marijn Koolen, editors, Proceedings of the 2nd Workshop on New Trends on Content-Based Recommender Systems co-located with 9th ACM Conference on Recommender Systems (RecSys 2015), Vienna, Austria, September 16-20, 2015., volume 1448 of CEUR Workshop Proceedings, pages 14–21. CEUR-WS.org, 2015.
  
For implicit feedback, LightFM has efficient implementation of BPR and WARP ranking losses and the corresponding optimization methods.

A Python implementation of LightFM is available on https://github.com/lyst/lightfm.

## Method 3: FM
Factorization Machine (FM) for context-aware recommendations.

> Steffen Rendle, Zeno Gantner, Christoph Freudenthaler, and Lars Schmidt-Thieme. Fast context-aware recommendations with factorization machines. In Proceedings of the 34th in- ternational ACM SIGIR conference on Research and development in Information Retrieval, pages 635–644. ACM, 2011.

> Steffen Rendle. Factorization machines with libfm. ACM Transactions on Intelligent Systems and Technology (TIST), 3(3):57, 2012.

The implementation of FM is available as [libFM](http://libfm.org). We can simply compile libFM and execute commends in terminal. The heaviest part of using this method is to carefully comply with the input data format and handle the output data.

Although FM is better with explicit feedback, we can still make use of it for implicit feedback by using binary classification and the bootstrap sampling strategy like BPR. Binary classification is an easy-to-use feature provided by libFM. Bootstrap sampling as data preprocessing needs to be done before using libFM to train the model. The commonly used way is to sample one negative item for each positive item under a given user-context configuration.

## Method 4: GPPW
Gaussian Process Factorization Machines (GPFM) is a non-linear context-aware collaborative filtering method that is based on Gaussian Processes. The variant method tailored for implicit feedback is GPPW (GPFM-based pairwise preference model).

> Trung V Nguyen, Alexandros Karatzoglou, and Linas Baltrunas. Gaussian process factoriz- ation machines for context-aware recommendations. In Proceedings of the 37th international ACM SIGIR conference on Research & development in information retrieval, pages 63–72. ACM, 2014.

The MATLAB implementation of GPFM is available at http://trungngv.github.io/gpfm.

To apply GPPW to the implicit feedback setting, we need to sample the negative/irrelevant feedback to each positive/relevant feedback and follow the input format required by the implementation of GPPW.

The input data required by different methods are summarized below.

|      | iALS | LightFM | FM | GPPW|
|------|------|---------|----|-----|
|input data | implicit feedback| binary feedback + metadata| binary feedback + context| binary feedback + context|

Check Chapter 3 of my thesis for detailed descussion and comparison of above four methods.  
