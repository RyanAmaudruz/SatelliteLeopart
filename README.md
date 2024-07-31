# SatelliteLeopart

This repository is part of the Thesis work: Sky’s the Limit: Satellite Imagery Analysis with Image-level and Dense Self-Supervised Techniques.

## Abstract
The introduction of the Vision Transformer (ViT) has revolutionized the field of computer
vision, significantly advancing research in self-supervised learning (SSL). While SSL devel-
opments have predominantly focused on object-centric and RGB images, the application of
these methods to satellite imagery poses unique challenges due to substantial domain shifts.
This study explores the use of a plain ViT backbone for satellite image analysis as it presents
multiple advantages over its hierarchical version.
We investigated three SSL frameworks — DINO, Leopart, and ODIN — evaluating their
performance on satellite images. Our findings indicate that pretraining on satellite images
provides a substantial advantage over object-centric RGB images, underscoring the value
of domain-specific pretraining. We observed that advanced dense SSL algorithms did not
consistently outperform traditional image-level SSL frameworks, with fine-tuning results
highlighting limitations in the dense approach when adapted to a ViT backbone. Furthermore,
linear probing performance did not reliably predict fine-tuning outcomes, suggesting that
linear probing may not fully reflect real-world application performance.
Notably, the plain ViT backbone, when combined with our selected SSL frameworks, learned
powerful representations that outperformed recent benchmarks on the DFC2020 and MADOS
datasets. Future research could enhance this framework by integrating a ViT-Adapter with the
ODIN algorithm to improve object detection granularity and training efficiency. This approach
could also enable the ViT backbone to process multiple data modalities, offering promising
potential for further advancements in SSL. Additionally, integrating a Mask2Former decoder
with the ViT-Backbone for semantic segmentation could further improve performance in
instance, panoptic, and semantic segmentation, making the model more general and robust.

Full paper available upon request.

Author: *Amaudruz R.*

Supervisors: *Yuki A., Russwurm, M.*

## Acknowledgements
This project builds upon the work in the repository [leopart](https://github.com/MkuuWaUjinga/leopart). The original research was conducted by Adrian Ziegler and Yuki M. Asano, as part of their study titled "[Self-supervised Learning of Object Parts for Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ziegler_Self-Supervised_Learning_of_Object_Parts_for_Semantic_Segmentation_CVPR_2022_paper.pdf)," 2022. We extend our gratitude to the authors of the work for their contributions to the open-source community, which have provided a valuable foundation for this project.

## SSL algorithm illustration
We show an illustration of the Leopart framework, taken from [leopart](https://github.com/MkuuWaUjinga/leopart):
![Alt Text](visuals/leopart_framework.png)

## Contributions
- [x] **Leopart Extension**: We create a new queuing strategy and add a DINO loss to allow the simultaneous supervision of both the spatial and global representations.
- [x] **Code refactoring**: We clean up and add comments to a few files.
- [x] **Satellite imagery**: We make the necessary adjustments to cater for Multi-Spectral images with 13 channels.

## Installation
We add a conda yaml file to set up the Python environment.

## Script
- [Satellite Leopart script](https://github.com/RyanAmaudruz/SatelliteLeopart/tree/main/snellius/finetune_with_leopart.sh): Script to run Satellite Leopart.

## License
This repository is released under the MIT License. The dataset and pretrained model weights are released under the CC-BY-4.0 license.
