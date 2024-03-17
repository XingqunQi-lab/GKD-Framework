# GKD-Framework
### Exploring Generalizable Distillation for Efficient Medical Image Segmentation

[Xingqun Qi](https://scholar.google.com.hk/citations?hl=zh-CN&user=3tO41a8AAAAJ&view_op=list_works&sortby=pubdate), [Zhuojie Wu](https://scholar.google.com.hk/citations?user=M2PRpz8AAAAJ&hl=zh-CN), [Wenxuan Zou](https://scholar.google.com.hk/citations?user=7Hskn8EAAAAJ&hl=zh-CN), [Min Ren](https://scholar.google.com.hk/citations?user=DQAHeWcAAAAJ&hl=zh-CN), [Muyi Sun](https://scholar.google.com.hk/citations?user=Ti7NNqMAAAAJ&hl=zh-CN), [Shanghang Zhang](https://scholar.google.com/citations?user=voqw10cAAAAJ&hl=en), [Caifeng Shan](https://scholar.google.com.hk/citations?user=fIXA_SsAAAAJ&hl=zh-CN), [Zhenan Sun](https://scholar.google.com.hk/citations?user=PuZGODYAAAAJ&hl=zh-CN)
### Pipeline
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/merged_framework.png)
The overall framework of our proposed Generalizable Knowledge Distillation (GKD). To be specific, in DCGD (left), each training sample is fed into two models simultaneously after different data augmentation strategies jointly. The MSAN encoder is exploited to extract domain-invariant knowledge.
We use the anchor vector and the augmented semantic vectors to construct two types of contrastive graphs for distillation. 
The pre-trained teacher network and MSAN structure are frozen when the student model is distilled during training. Besides, in DICD (right), the model-specific features from teacher and student are cross-reconstructed by the header exchanging of MSAN. The pre-trained teacher model is utilized to provide the cross distillation supervision on both the encoder and decoder of the student respectively.
### tSNE Visualization
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/tSNE.png)
The t-SNE visualization on the latent vectors of the teacher and student networks with random samples in the CHASEDB1 dataset. The different colors denote anchor samples and augmented samples. Once our proposed GKD framework distills the student, the latent vectors demonstrate noticeable distribution discrepancies compared with scratch-trained ones
### Results on Cross-domain Datasets
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/vessel_results_detail.png)
Retinal vessel segmentation comparisons of the proposed GKD with previous knowledge distillation counterparts. 
The top two-row samples are from the CHASEDB1 dataset, the middle two-row samples are from the STARE dataset, and the bottom two-row samples are from the DRIVE dataset.
Notably, all the models are trained on the CHASEDB1 dataset and directly tested on the three testing sets.
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/liver_results.png)
Liver segmentation comparisons of the proposed GKD with state-of-the-art knowledge distillation counterparts. 
The top two-row samples are from the CHAOS dataset, and the bottom two-row samples are from the LITS dataset.
### Results of Ablation Study
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/vessel_ablation_study_detail.png)
The visual comparisons of ablation study on the retinal vessel segmentation task. 
The top two-row samples are from the CHASEDB1 dataset, the middle two-row samples are from the STARE dataset, and the bottom two-row samples are from the DRIVE dataset.
Notably, all the models are trained on the CHASEDB1 dataset and directly tested on three datasets.
![image](https://github.com/XingqunQi-lab/GKD-Framework/blob/main/image/liver_ablation_study_detail.png)
The visual comparisons of ablation study on the liver segmentation task. 
The top two-row samples are from CHAOS dataset, and the bottom two-row samples are from the LITS dataset. 
Notably, all the models are trained on the CHAOS dataset and directly tested on these two datasets.

# Code
We are organizing our code and will release the full version code soon. Sorry for the delay.
