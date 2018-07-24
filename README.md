# Introduction

This repository includes the main code part of my master thesis "*Implicit Feedback Based Context-Aware Recommender For Music Light System*". 

https://pure.tue.nl/ws/files/90854618/master_thesis_final_revised_SiqiLi.pdf

# DataSet
The dataset used is [Frappe](http://baltrunas.info/research-menu/frappe). 

File [frappe.ipynb](/frappe.ipynb) contains the code to examine the dataset and to prepare for later expariments.  

# Evaluation Metrics
Since implicit feedback based context-aware recommendation is a top-N recommendation task, I basically use four list based metrics: precision@N, MAP@N, NDCG@N and ERR@N, where N = 10 considering the size of the Frappe dataset. Precision and MAP are two classification accuracy metrics and NDCG and ERR are two ranking accuracy metrics; see Section 2.5.2 of my thesis. Note that since precision is the mean precision of recommendations for all user-context configuration, I refer to MP. Scores by all of these four metrics are the higher the better.

 - MP: Mean Precision
 - MAP: Mean Average Precision
 - NDCG: Normalized Discounted Cumulative Gain
 - ERR: Expected Reciprocal Rank

Check Section 2.5 and 4.3.2 of my thesis for detailed descussion and comparison of above four metrics and other evaluation metrics recommender systems. 

File measure.py is the implementation of above four metrics.

*From chapter 8 of Andrew NG's new book [*Machine Learning Yearning*](http://www.mlyearning.org), one single-number evaluation metric is preferred for the team to optimize in practice. If we are care multiple metrics, we can simply take the average of them. 

# Methods
Four start-of-the-art methods are studied with experiments. The focus is on the two context-aware recommender methods: FM and GPPW.
  - iALS: the matrix factorization(MF) method specially for handling data with implicit feedback.
  
(*Yifan Hu, Yehuda Koren, and Chris Volinsky. Collaborative filtering for implicit feedback datasets. In Data Mining, 2008. ICDM’08. Eighth IEEE International Conference on, pages 263–272. Ieee, 2008.*)
  
  - LightFM: a hybrid MF model representing users and items as linear combinations of their context features’ latent factors. For implicit feedback, LightFM has efficient implementation of BPR and WARP ranking losses and the corresponding optimization methods.
  
  A Python implementation of LightFM is available on https://github.com/lyst/lightfm.

  - FM:
  
  - GPPW:
  
  Check Chapter 3 of thesis for detailed descussion and comparison of above four methods.  
