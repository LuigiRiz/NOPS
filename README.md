# **Novel Class Discovery for 3D Point Cloud Semantic Segmentation [CVPR 2023]**
The official implementation of our work "Novel Class Discovery for 3D Point Cloud Semantic Segmentation".

![teaser](assets/NOPS_teaser.jpg)

## Introduction
Novel class discovery (NCD) for semantic segmentation is the problem of learning a model that is capable of segmenting unlabelled (novel) classes by using only the supervision from labelled (base) classes.
This problem has been recently pioneered for 2D image data, but no work exists for 3D point cloud data.
In fact, assumptions made for 2D are loosely applicable to 3D in this case.
This paper is thus presented to advance the state of the art on point cloud data analysis in four directions.
Firstly, we address the new problem of NCD for point cloud semantic segmentation.
Secondly, we show that the transposition of the only existing NCD method for 2D semantic segmentation to 3D data is suboptimal.
Thirdly, we present a new method for NCD based on online clustering that exploits uncertainty quantification to produce prototypes for pseudo-labelling the points of the novel classes.
Lastly, we introduce a new evaluation protocol to assess the performance of NCD for point cloud semantic segmentation.
We thoroughly evaluate our method on SemanticKITTI and SemanticPOSS datasets, showing that it can significantly outperform the baseline.

Camera ready and code will be released soon!
