# This repo supplements our [3D Vision with Transformers Survey](https://arxiv.org/abs/2208.04309)
Jean Lahoud, Jiale Cao, Fahad Shahbaz Khan, Hisham Cholakkal, Rao Muhammad Anwer, Salman Khan, Ming-Hsuan Yang

This repo includes all the 3D computer vision papers with Transformers which are presented in our [paper](https://arxiv.org/abs/2208.04309), and we aim to frequently update the latest relevant papers.

<p align="center">
<img src="https://user-images.githubusercontent.com/14073587/183882596-ada49e17-bbd5-4b09-962b-e0ff1d8291c0.png" width="600">
</p>

#### Content
- [Object Classification](#object-classification)<br>
- [3D Object Detection](#3d-object-detection)<br>
- [3D Segmentation](#3d-segmentation)<br>
  - [Complete Scenes Segmentation](#complete-scenes-segmentation)<br>
  - [Point Cloud Video Segmentation](#point-cloud-video-segmentation)<br>
  - [Medical Imaging Segmentation](#medical-imaging-segmentation)<br>
- [3D Point Cloud Completion](#3d-point-cloud-completion)<br>
- [3D Pose Estimation](#3d-pose-estimation)<br>
- [Other Tasks](#other-tasks)<br>
  - [3D Tracking](#3d-tracking)<br>
  - [3D Motion Prediction](#3d-motion-prediction)<br>
  - [3D Reconstruction](#3d-reconstruction)<br>
  - [Point Cloud Registration](#point-cloud-registration)<br>



## Object Classification
Group-in-Group Relation-Based Transformer for 3D Point Cloud Learning [**RS 2022**][[PDF](https://www.mdpi.com/2072-4292/14/7/1563/pdf?version=1648109597 )] <br>

Masked Autoencoders for Point Cloud Self-supervised Learning [**ECCV 2022**][[PDF](https://arxiv.org/pdf/2203.06604)][[Code](https://github.com/Pang-Yatian/Point-MAE )] <br>

3DCTN: 3D Convolution-Transformer Network for Point Cloud Classification [**T-ITS 2022**][[PDF](https://arxiv.org/pdf/2203.00828 )] <br>

LFT-Net: Local Feature Transformer Network for Point Clouds Analysis [**T-ITS 2022**][[PDF](https://ieeexplore.ieee.org/document/9700748/ )] <br>

Sewer defect detection from 3D point clouds using a transformer-based deep learning model [**Automation in Construction 2022**][[PDF](https://www.mdpi.com/1424-8220/22/12/4517/pdf?version=1655277701 )] <br>

3d medical point transformer: Introducing convolution to attention networks for medical point cloud analysis [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2112.04863)][[Code](https://github.com/crane-papercode/3dmedpt )] <br>

Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_Point-BERT_Pre-Training_3D_Point_Cloud_Transformers_With_Masked_Point_Modeling_CVPR_2022_paper.pdf)][[Code](https://github.com/lulutang0608/Point-BERT )] <br>

CpT: Convolutional Point Transformer for 3D Point Cloud Processing [**ACCVW 2022**][[PDF](https://arxiv.org/pdf/2111.10866 )] <br>

PatchFormer: An Efficient Point Transformer With Patch Attention [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PatchFormer_An_Efficient_Point_Transformer_With_Patch_Attention_CVPR_2022_paper.pdf)] <br>

PVT: Point-Voxel Transformer for Point Cloud Learning [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2108.06076.pdf)][[Code](https://github.com/HaochengWan/PVT )] <br>

Adaptive Wavelet Transformer Network for 3D Shape Representation Learning [**ICLR 2021**][[PDF](https://openreview.net/pdf?id=5MLb3cLCJY )] <br>

Point cloud learning with transformer [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2104.13636 )] <br>

3crossnet: Cross-level cross-scale cross-attention network for point cloud representation [**RA-L 2022**][[PDF](https://arxiv.org/pdf/2104.13053 )] <br>

Dual Transformer for Point Cloud Analysis [**IEEE Trans Multimedia**][[PDF](https://arxiv.org/pdf/2104.13044 )] <br>

Centroid transformers: Learning to abstract with attention [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2102.08606 )] <br>

PCT: Point cloud transformer [**CVPR 2019**][[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf)][[Code](https://github.com/MenghaoGuo/PCT )] <br>

Point Transformer [**ICCV 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)][[Code](https://github.com/POSTECH-CVLab/point-transformer )] <br>

Point Transformer [**IEEE Access 2021**][[PDF](https://arxiv.org/pdf/2011.00931)][[Code](https://github.com/engelnico/point-transformer )] <br>

Modeling point clouds with self-attention and gumbel subset sampling [**CVPR 2019**][[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf )] <br>

Attentional shapecontextnet for point cloud recognition [**CVPR 2018**][[PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xie_Attentional_ShapeContextNet_for_CVPR_2018_paper.pdf)][[Code](https://github.com/umyta/A-SCN )] <br>

## 3D Object Detection

NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields [**ECCV 2024**][[PDF](https://arxiv.org/pdf/2404.01300)] [[WebPage](https://nerf-mae.github.io/)] [[Code](https://github.com/zubair-irshad/NeRF-MAE)] <be>

Bridged Transformer for Vision and Point Cloud 3D Object Detection [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Bridged_Transformer_for_Vision_and_Point_Cloud_3D_Object_Detection_CVPR_2022_paper.pdf )] <br>

Multimodal Token Fusion for Vision Transformers [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.pdf )][[Code](https://github.com/yikaiw/TokenFusion )] <br>

CAT-Det: Contrastively Augmented Transformer for Multi-modal 3D Object Detection [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_CAT-Det_Contrastively_Augmented_Transformer_for_Multi-Modal_3D_Object_Detection_CVPR_2022_paper.pdf )] <br>

Focused Decoding Enables 3D Anatomical Detection by Transformers [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2207.10774.pdf)][[Code](https://github.com/bwittmann/transoar )] <br>

MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2203.13310)][[Code](https://github.com/ZrrSkywalker/MonoDETR )] <br>

TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.pdf)][[Code](https://github.com/XuyangBai/TransFusion )] <br>

Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Voxel_Set_Transformer_A_Set-to-Set_Approach_to_3D_Object_Detection_CVPR_2022_paper.pdf)][[Code](https://github.com/skyhehe123/VoxSeT )] <br>

VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_VISTA_Boosting_3D_Object_Detection_via_Dual_Cross-VIew_SpaTial_Attention_CVPR_2022_paper.pdf)][[Code](https://github.com/Gorilla-Lab-SCUT/VISTA )] <br>

Point Density-Aware Voxels for LiDAR 3D Object Detection [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Hu_Point_Density-Aware_Voxels_for_LiDAR_3D_Object_Detection_CVPR_2022_paper.pdf)][[Code](https://github.com/TRAILab/PDV )] <br>

PETR: Position Embedding Transformation for Multi-View 3D Object Detection [**ECCV 2022**][[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136870523.pdf)][[Code](https://github.com/megvii-research/PETR )] <br>

ARM3D: Attention-based relation module for indoor 3D object detection [**Comput. Vis.**][[PDF](https://link.springer.com/content/pdf/10.1007/s41095-021-0252-6.pdf)][[Code](https://github.com/lanlan96/arm3d )] <br>

MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_MonoDTR_Monocular_3D_Object_Detection_With_Depth-Aware_Transformer_CVPR_2022_paper.pdf)][[Code](https://github.com/KuanchihHuang/MonoDTR )] <br>

Attention-based Proposals Refinement for 3D Object Detection [**IV 2022**][[PDF](https://arxiv.org/pdf/2201.07070)][[Code](https://github.com/quan-dao/APRO3D-Net )] <br>

Embracing Single Stride 3D Object Detector with Sparse Transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_Embracing_Single_Stride_3D_Object_Detector_With_Sparse_Transformer_CVPR_2022_paper.pdf)][[Code](https://github.com/tusen-ai/SST )] <br>

Fast Point Transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Fast_Point_Transformer_CVPR_2022_paper.pdf)][[Code](https://github.com/POSTECH-CVLab/FastPointTransformer )] <br>

BoxeR: Box-Attention for 2D and 3D Transformers [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Nguyen_BoxeR_Box-Attention_for_2D_and_3D_Transformers_CVPR_2022_paper.pdf)][[Code](https://github.com/kienduynguyen/BoxeR )] <br>

DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries [**CoRL 2022**][[PDF](https://proceedings.mlr.press/v164/wang22b/wang22b.pdf)][[Code](https://github.com/WangYueFt/detr3d )] <br>

An End-to-End Transformer Model for 3D Object Detection [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Misra_An_End-to-End_Transformer_Model_for_3D_Object_Detection_ICCV_2021_paper.pdf)][[Code](https://github.com/facebookresearch/3detr )] <br>

Voxel Transformer for 3D Object Detection [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Mao_Voxel_Transformer_for_3D_Object_Detection_ICCV_2021_paper.pdf)][[Code](https://github.com/PointsCoder/VOTR )] <br>

Improving 3D Object Detection with Channel-wise Transformer [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Sheng_Improving_3D_Object_Detection_With_Channel-Wise_Transformer_ICCV_2021_paper.pdf)][[Code](https://github.com/hlsheng1/CT3D )] <br>

M3DETR: Multi-representation, Multi-scale, Mutual-relation 3D Object Detection with Transformers [**WACV 2022**][[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Guan_M3DETR_Multi-Representation_Multi-Scale_Mutual-Relation_3D_Object_Detection_With_Transformers_WACV_2022_paper.pdf)][[Code](https://github.com/rayguan97/M3DeTR )] <br>

Group-Free 3D Object Detection via Transformers [**ICCV 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Group-Free_3D_Object_Detection_via_Transformers_ICCV_2021_paper.pdf)][[Code](https://github.com/zeliu98/Group-Free-3D )] <br>

SA-Det3D: Self-Attention Based Context-Aware 3D Object Detection [**ICCVW 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021W/AVVision/papers/Bhattacharyya_SA-Det3D_Self-Attention_Based_Context-Aware_3D_Object_Detection_ICCVW_2021_paper.pdf)][[Code](https://github.com/AutoVision-cloud/SA-Det3D )] <br>

3D object detection with pointformer [**CVPR 2021**][[PDF](http://openaccess.thecvf.com/content/CVPR2021/papers/Pan_3D_Object_Detection_With_Pointformer_CVPR_2021_paper.pdf)][[Code](https://github.com/Vladimir2506/Pointformer )] <br>

Temporal-Channel Transformer for 3D Lidar-Based Video Object Detection in Autonomous Driving [**IEEE Trans. Circuits Syst.**][[PDF](https://arxiv.org/pdf/2011.13628 )] <br>

MLCVNet: Multi-Level Context VoteNet for 3D Object Detection [**CVPR 2020**][[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_MLCVNet_Multi-Level_Context_VoteNet_for_3D_Object_Detection_CVPR_2020_paper.pdf)][[Code](https://github.com/NUAAXQ/MLCVNet )] <br>

LiDAR-based online 3d video object detection with graph-based message passing and spatiotemporal transformer attention [**CVPR 2020**][[PDF](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_LiDAR-Based_Online_3D_Video_Object_Detection_With_Graph-Based_Message_Passing_CVPR_2020_paper.pdf)][[Code](https://github.com/yinjunbo/3DVID )] <br>

SCANet: Spatial-channel attention network for 3d object detection [**ICASSP 2019**][[PDF](https://ieeexplore.ieee.org/document/8682746)][[Code](https://github.com/zhouruqin/SCANet )] <br>

## 3D Segmentation
For part segmentation, check [Object Classification](#object-classification)

#### Complete Scenes Segmentation

Stratified Transformer for 3D Point Cloud Segmentation [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)][[Code](https://github.com/dvlab-research/Stratified-Transformer )] <br>

Multimodal Token Fusion for Vision Transformers [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Multimodal_Token_Fusion_for_Vision_Transformers_CVPR_2022_paper.pdf )][[Code](https://github.com/yikaiw/TokenFusion )] <br>

Sparse Cross-scale Attention Network for Efficient LiDAR Panoptic Segmentation [**AAAI 2022**][[PDF](https://www.aaai.org/AAAI22Papers/AAAI-5976.XuS.pdf )] <br>

Fast Point Transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_Fast_Point_Transformer_CVPR_2022_paper.pdf)][[Code](https://github.com/POSTECH-CVLab/FastPointTransformer )] <br>

Segment-Fusion: Hierarchical Context Fusion for Robust 3D Semantic Segmentation [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Thyagharajan_Segment-Fusion_Hierarchical_Context_Fusion_for_Robust_3D_Semantic_Segmentation_CVPR_2022_paper.pdf )] <br>

#### Point Cloud Video Segmentation

Point Spatio-Temporal Transformer Networks for Point Cloud Video Modeling [**TPAMI 2022**][[PDF](https://ieeexplore.ieee.org/abstract/document/9740525 )] <br>

Spatial-Temporal Transformer for 3D Point Cloud Sequences [**WACV 2022**][[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Wei_Spatial-Temporal_Transformer_for_3D_Point_Cloud_Sequences_WACV_2022_paper.pdf )] <br>

Point 4D transformer networks for spatio-temporal modeling in point cloud videos [**CVPR 2021**][[PDF](http://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf)][[Code](https://github.com/hehefan/P4Transformer )] <br>

#### Medical Imaging Segmentation
Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images [**MICCAI BrainLes 2022**][[PDF](https://arxiv.org/pdf/2201.01266)][[Code](https://github.com/Project-MONAI/research-contributions/tree/master/SwinUNETR/BRATS21 )] <br>

D-Former: A U-shaped Dilated Transformer for 3D Medical Image Segmentation [**Neural Comput Appl 2022**][[PDF](https://arxiv.org/pdf/2201.00462 )] <br>

A Robust Volumetric Transformer for Accurate 3D Tumor Segmentation [**MICCAI 2022**][[PDF](https://arxiv.org/pdf/2111.13300)][[Code](https://github.com/himashi92/VT-UNet )] <br>

T-AutoML: Automated Machine Learning for Lesion Segmentation using Transformers in 3D Medical Imaging [**ICCV 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_T-AutoML_Automated_Machine_Learning_for_Lesion_Segmentation_Using_Transformers_in_ICCV_2021_paper.pdf )] <br>

After-unet: Axial fusion transformer unet for medical image segmentation [**WACV 2022**][[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Yan_AFTer-UNet_Axial_Fusion_Transformer_UNet_for_Medical_Image_Segmentation_WACV_2022_paper.pdf )] <br>

Bitr-unet: a cnn-transformer combined network for mri brain tumor segmentation [**MICCAI BrainLes 2022**][[PDF](https://arxiv.org/pdf/2109.12271 )] <br>

nnformer: Interleaved transformer for volumetric segmentation [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2109.03201)][[Code](https://github.com/282857341/nnFormer )] <br>

UTNet: A Hybrid Transformer Architecture for Medical Image Segmentation [**MICCAI 2022**][[PDF](https://arxiv.org/abs/2107.00781)][[Code](https://github.com/yhygao/UTNet )] <br>

Medical image segmentation using squeezeand-expansion transformers [**IJCAI 2021**][[PDF](https://arxiv.org/pdf/2105.09511)][[Code](https://github.com/askerlee/segtran )] <br>

Unetr: Transformers for 3d medical image segmentation [**WACV 2022**][[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf)][[Code](https://github.com/Project-MONAI/research-contributions/tree/master/UNETR/BTCV )] <br>

Transbts: Multimodal brain tumor segmentation using transformer [**MICCAI 2021**][[PDF](https://arxiv.org/pdf/2103.04430)][[Code](https://github.com/Wenxuan-1119/TransBTS )] <br>

Spectr: Spectral transformer for hyperspectral pathology image segmentation [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2103.03604)][[Code](https://github.com/hfut-xc-yun/SpecTr )] <br>

Cotr: Efficiently bridging cnn and transformer for 3d medical image segmentation [**MICCAI 2021**][[PDF](https://arxiv.org/pdf/2103.03024)][[Code](https://github.com/YtongXie/CoTr )] <br>

Convolution-free medical image segmentation using transformers [**MICCAI 2021**][[PDF](https://arxiv.org/pdf/2102.13645 )] <br>

Transfuse: Fusing transformers and cnns for medical image segmentation [**MICCAI 2021**][[PDF](https://arxiv.org/pdf/2102.08005)][[Code](https://github.com/Rayicer/TransFuse )] <br>

## 3D Point Cloud Completion
Learning Local Displacements for Point Cloud Completion [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learning_Local_Displacements_for_Point_Cloud_Completion_CVPR_2022_paper.pdf)][[Code](https://github.com/wangyida/disp3d )] <br>

AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Mittal_AutoSDF_Shape_Priors_for_3D_Completion_Reconstruction_and_Generation_CVPR_2022_paper.pdf)][[Code](https://github.com/yccyenchicheng/AutoSDF )] <br>

PointAttN: You Only Need Attention for Point Cloud Completion [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2203.08485)][[Code](https://github.com/ohhhyeahhh/PointAttN )] <br>

Point cloud completion on structured feature map with feedback network [**CVM 2022**][[PDF](https://arxiv.org/pdf/2202.08583 )] <br>

ShapeFormer: Transformer-based Shape Completion via Sparse Representation [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yan_ShapeFormer_Transformer-Based_Shape_Completion_via_Sparse_Representation_CVPR_2022_paper.pdf)][[Code](https://github.com/QhelDIV/ShapeFormer )] <br>

A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion [**ICLR 2021**][[PDF](https://arxiv.org/pdf/2112.03530)][[Code](https://github.com/ZhaoyangLyu/Point_Diffusion_Refinement )] <br>

MFM-Net: Unpaired Shape Completion Network with Multi-stage Feature Matching [**arXiv 2021**][[PDF](https://arxiv.org/pdf/2111.11976 )] <br>

PCTMA-Net: Point Cloud Transformer with Morphing Atlas-based Point Generation Network for Dense Point Cloud Completion [**IROS 2021**][[PDF](https://www.researchgate.net/profile/Alexander-Perzylo/publication/353955048_PCTMA-Net_Point_Cloud_Transformer_with_Morphing_Atlas-based_Point_Generation_Network_for_Dense_Point_Cloud_Completion/links/611bd6930c2bfa282a50001d/PCTMA-Net-Point-Cloud-Transformer-with-Morphing-Atlas-based-Point-Generation-Network-for-Dense-Point-Cloud-Completion.pdf)][[Code](https://github.com/LinJianjie/PCTMA_Net )] <br>

PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.pdf)][[Code](https://github.com/yuxumin/PoinTr )] <br>

SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Xiang_SnowflakeNet_Point_Cloud_Completion_by_Snowflake_Point_Deconvolution_With_Skip-Transformer_ICCV_2021_paper.pdf)][[Code](https://github.com/AllenXiangX/SnowflakeNet )] <br>

## 3D Pose Estimation
Permutation-Invariant Relational Network for Multi-person 3D Pose Estimation [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2204.04913 )] <br>

Zero-Shot Category-Level Object Pose Estimation [**ECCV 2022**][[PDF](https://arxiv.org/pdf/2204.03635)][[Code](https://github.com/applied-ai-lab/zero-shot-pose )] <br>

Efficient Virtual View Selection for 3D Hand Pose Estimation [**AAAI 2022**][[PDF](https://www.aaai.org/AAAI22Papers/AAAI-1352.ChengJ.pdf)][[Code](https://github.com/iscas3dv/handpose-virtualview )] <br>

Learning-based Point Cloud Registration for 6D Object Pose Estimation in the Real World [**ECCV 2022**][[PDF](https://infoscience.epfl.ch/record/295132/files/ECCV2022_Match_Normalisation_Point_Cloud_Registration__New_.pdf)][[Code](https://github.com/dangzheng/matchnorm )] <br>

CrossFormer: Cross Spatio-Temporal Transformer for 3D Human Pose Estimation [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2203.13387)][[Code](https://github.com/mfawzy/CrossFormer )] <br>

RayTran: 3D pose estimation and shape reconstruction of multiple objects from videos with ray-traced transformers [**ECCV 2022**][[PDF](https://arxiv.org/pdf/2203.13296 )] <br>

P-STMO: Pre-Trained Spatial Temporal Many-to-One Model for 3D Human Pose Estimation [**ECCV 2022**][[PDF](https://arxiv.org/pdf/2203.07628)][[Code](https://github.com/paTRICK-swk/P-STMO )] <br>

MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_MixSTE_Seq2seq_Mixed_Spatio-Temporal_Encoder_for_3D_Human_Pose_Estimation_CVPR_2022_paper.pdf)][[Code](https://github.com/JinluZhang1126/MixSTE )] <br>

6D-ViT: Category-Level 6D Object Pose Estimation via Transformer-based Instance Representation Learning [**TIP 2022**][[PDF](https://arxiv.org/pdf/2110.04792 )] <br>

Keypoint Transformer: Solving Joint Identification in Challenging Hands and Object Interactions for Accurate 3D Pose Estimation [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Hampali_Keypoint_Transformer_Solving_Joint_Identification_in_Challenging_Hands_and_Object_CVPR_2022_paper.pdf)][[Code](https://github.com/shreyashampali/kypt_transformer )] <br>

Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose Estimation [**IEEE Trans. Multimed. 2022**][[PDF](https://arxiv.org/pdf/2103.14304)][[Code](https://github.com/Vegetebird/StridedTransformer-Pose3D )] <br>

3D Human Pose Estimation with Spatial and Temporal Transformers [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_3D_Human_Pose_Estimation_With_Spatial_and_Temporal_Transformers_ICCV_2021_paper.pdf)][[Code](https://github.com/zczcwh/PoseFormer )] <br>

End-to-End Human Pose and Mesh Reconstruction with Transformers [**CVPR 2021**][[PDF](http://openaccess.thecvf.com/content/CVPR2021/papers/Lin_End-to-End_Human_Pose_and_Mesh_Reconstruction_with_Transformers_CVPR_2021_paper.pdf)][[Code](https://github.com/microsoft/MeshTransformer )] <br>

PI-Net: Pose Interacting Network for Multi-Person Monocular 3D Pose Estimation [**WACV 2021**][[PDF](http://openaccess.thecvf.com/content/WACV2021/papers/Guo_PI-Net_Pose_Interacting_Network_for_Multi-Person_Monocular_3D_Pose_Estimation_WACV_2021_paper.pdf)][[Code](https://github.com/GUO-W/PI-Net )] <br>

HOT-Net: Non-Autoregressive Transformer for 3D Hand-Object Pose Estimation [**ACM MM 2020**][[PDF](https://dl.acm.org/doi/pdf/10.1145/3394171.3413775 )] <br>

Hand-Transformer: Non-Autoregressive Structured Modeling for 3D Hand Pose Estimation [**ECCV 2020**][[PDF](https://cse.buffalo.edu/~jsyuan/papers/2020/4836.pdf )] <br>

Epipolar Transformer for Multi-view Human Pose Estimation [**CVPRW 2020**][[PDF](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w70/He_Epipolar_Transformer_for_Multi-View_Human_Pose_Estimation_CVPRW_2020_paper.pdf)][[Code](https://github.com/yihui-he/epipolar-transformers )] <br>

## Other Tasks

#### 3D Tracking
Pttr: Relational 3d point cloud object tracking with transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_PTTR_Relational_3D_Point_Cloud_Object_Tracking_With_Transformer_CVPR_2022_paper.pdf)][[Code](https://github.com/Jasonkks/PTTR  )] <br>

3d object tracking with transformer [**BMVC 2021**][[PDF](https://arxiv.org/pdf/2110.14921 )] <br>

#### 3D Motion Prediction
Hr-stan: High-resolution spatio-temporal attention network for 3d human motion prediction [**CVPRW 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022W/Precognition/papers/Medjaouri_HR-STAN_High-Resolution_Spatio-Temporal_Attention_Network_for_3D_Human_Motion_Prediction_CVPRW_2022_paper.pdf )] <br>

Gimo: Gaze-informed human motion prediction in context [**ECCV 2022**][[PDF](https://arxiv.org/pdf/2204.09443)][[Code](https://github.com/y-zheng18/GIMO )] <br>

Pose transformers (potr): Human motion prediction with non-autoregressive transformer [**ICCVW 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021W/SoMoF/papers/Martinez-Gonzalez_Pose_Transformers_POTR_Human_Motion_Prediction_With_Non-Autoregressive_Transformers_ICCVW_2021_paper.pdf)][[Code](https://github.com/idiap/potr)] <br>

Learning progressive joint propagation for human motion prediction [**ECCV 2020**][[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520222.pdf )] <br>

History repeats itself: Human motion prediction via motion attention [**ECCV 2020**][[PDF](https://arxiv.org/pdf/2007.11755)][[Code](https://github.com/wei-mao-2019/HisRepItself )] <br>

A spatio-temporal transformer for 3d human motion prediction [**3DV 2021**][[PDF](https://arxiv.org/pdf/2004.08692)][[Code](https://github.com/eth-ait/motion-transformer )] <br>

#### 3D Reconstruction
Vpfusion: Joint 3d volume and pixel-aligned feature fusion for single and multi-view 3d reconstruction [**arXiv 2022**][[PDF](https://arxiv.org/pdf/2203.07553 )] <br>

Thundr: Transformer-based 3d human reconstruction with marker [**ICCV 2021**][[PDF](http://openaccess.thecvf.com/content/ICCV2021/papers/Zanfir_THUNDR_Transformer-Based_3D_Human_Reconstruction_With_Markers_ICCV_2021_paper.pdf )] <br>

Multi-view 3d reconstruction with transformer [**ICCV 2021**][[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Multi-View_3D_Reconstruction_With_Transformers_ICCV_2021_paper.pdf )] <br>

#### Point Cloud Registration
Regtr: End-to-end point cloud correspondences with transformer [**CVPR 2022**][[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Yew_REGTR_End-to-End_Point_Cloud_Correspondences_With_Transformers_CVPR_2022_paper.pdf)][[Code](https://github.com/yewzijian/RegTR )] <br>

LegoFormer: Transformers for Block-by-Block Multi-view 3D Reconstruction [**CVPR 2021**][[PDF](https://arxiv.org/abs/2106.12102)][[Code](https://github.com/faridyagubbayli/LegoFormer )] <br>

Robust point cloud registra tion framework based on deep graph matching [**CVPR 2021**][[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Robust_Point_Cloud_Registration_Framework_Based_on_Deep_Graph_Matching_CVPR_2021_paper.pdf)][[Code](https://github.com/fukexue/RGM )] <br>

Deep closest point: Learning representations for point cloud registration [**ICCV 2019**][[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Deep_Closest_Point_Learning_Representations_for_Point_Cloud_Registration_ICCV_2019_paper.pdf)][[Code](https://github.com/WangYueFt/dcp )] <br>




# Citation

If you find the listing or the survey useful for your work, please cite our paper:

```
@misc{lahoud20223d,
      title={3D Vision with Transformers: A Survey}, 
      author={Lahoud, Jean and Cao, Jiale and Khan, Fahad Shahbaz and Cholakkal, Hisham and Anwer, Rao Muhammad and Khan, Salman and Yang, Ming-Hsuan},
      year={2022},
      eprint={2208.04309},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
