# ReBADD-SE (under construction)
This is the repository for ReBADD-SE, a multi-objective molecular optimization model that designs a molecular structures in the format of SELFIES.
For more details, please refer to our [paper](https://www.sciencedirect.com/science/article/pii/S0010482523001865).

**Notice** We are currently in the process of updating the repository to enhance the usability of ReBADD-SE. We apologize for any inconvenience this may cause during the update. The update is expected to be completed within this week, and we appreciate your patience. 

- Latest update: 24 Jan 2024



Install
----
```
conda env create -f environment.yml
```



Task Descriptions
----
- TASK1: ReBADD-SE for GSK3b, JNK3, QED, and SA (frag-level)
- TASK3: ReBADD-SE for BCL2, BCLXL, and BCLW (frag-level)
- TASK4: ReBADD-SE for BCL2, BCLXL, and BCLW (char-level)
- TASK7: SELFIES Collapse Analaysis between ReBADD-SE (frag, char-level) and GA+D



Notebook Descriptions
----
## 0_preprocess_data.ipynb
- Read the training data
- Preprocess the data for model training
- The preprocessed data are stored in the 'processed_data' directory

## 1_pretraining.ipynb
- Read the training data
- The generator learns the grammar rules of SELFIES

## 2_optimize+{objectives}.ipynb
- (Important!) Please check first the 'ReBADD_config.py' in which a reward function have to be defined appropriately
- Load the pretrained generator

## 3_checkpoints+{objectives}.ipynb
- Load the checkpoints stored during optimization
- Sample molecules for each checkpoint

## 4_calculate_properties.ipynb
- For each checkpoint, load the sampled molecules
- Evaluate their property scores

## 5_evaluate_checkpoints.ipynb
- Calculate metrics (e.g. success rate)
- Find the best checkpoint



Note
----
If you have any further questions, please do not hesitate to let me know.
```
jonghwanc@hallym.ac.kr
```


Citation
----
```
@article{CHOI2023106721,
	title = {ReBADD-SE: Multi-objective molecular optimisation using SELFIES fragment and off-policy self-critical sequence training},
	journal = {Computers in Biology and Medicine},
	volume = {157},
	pages = {106721},
	year = {2023},
	issn = {0010-4825},
	doi = {https://doi.org/10.1016/j.compbiomed.2023.106721},
	url = {https://www.sciencedirect.com/science/article/pii/S0010482523001865},
	author = {Jonghwan Choi and Sangmin Seo and Seungyeon Choi and Shengmin Piao and Chihyun Park and Sung Jin Ryu and Byung Ju Kim and Sanghyun Park},
	keywords = {Drug discovery, De novo drug design, Multi-objective optimisation, SELFIES, Reinforcement learning}
}
```