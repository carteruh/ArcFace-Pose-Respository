# Minimizing Number of Distinct Poses for Pose-Invariant Face Recognition
**[Carter Ung](https://github.com/carteruh), [Pranav Mantini](https://www2.cs.uh.edu/~pmantini/), [Shishir Shah](https://www.ee.uh.edu/faculty/shah)**

<img src="framework.png" alt="teaser" width="800"/>

## Summary 
In this study, we propose a training framework to enhance pose-invariant face recognition by identifying the minimum number of poses for training deep convolutional neural network (CNN) models, enabling recognition across large pose variations in pitch (nodding) and yaw (shaking) rotation angles. We deploy ArcFace, a state-of-the-art recognition model, to evaluate model performance in a probe-gallery matching task across groups of facial poses categorized by pitch and yaw Euler angles. Our experiments train and assess ArcFace on varying pose bins to determine the top-1 accuracy and observe how recognition accuracy is affected. Our findings reveal that: (i) a group of poses at -45°, 0°, and +45° yaw angles achieve uniform top-1 accuracy across all yaw poses, (ii) recognition performance is better with negative pitch angles than positive pitch angles, and (iii) training with image augmentations like horizontal flips results in similar or better performance, further minimizing yaw poses to a frontal and 3/4 view.

We use improve residual networks (i.e iresnet50) CNN backbones to evaluate the face recognition rates of the ArcFace margin-penalized CNN-based model on different pose groups. The goal of this research is to study the relationships between pose groups and how the embeddings of each pose group can be generalized across different pose ranges. The focus of the pose groups will be based on pitch and yaw variations of the face. 

## Citation


```
@inproceedings{ung2025posefr,
    title={Minimizing Number of Distinct Poses for Pose-Invariant Face Recognition}, 
    author={Carter Ung, Pranav Mantini, Shishir Shah},
    booktitle={VISAPP},
    year={2025}
}
```


## Acknowledgements

This empirical study is conducted by utilizing `ArcFace` for model evaluation and the `M2FPA` dataset to analyze various facial poses.

+ [[ICCV 2019] M2FPA: A Multi-Yaw Multi-Pitch High-Quality Database and Benchmark for Facial Pose Analysis](https://pp2li.github.io/M2FPA-dataset/)
+ [[CVPR 2019] ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://insightface.ai/arcface)
