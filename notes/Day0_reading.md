# Day 0 Reading Notes – ECE 595 Term Paper

## 1. Liu et al. (2023) – Rectified Flow
**Core Idea:**  
**Key Equations:**  
**Main Strengths:** 
**Limitations / Open Questions:**  
**Connection to My Paper:**  

## 2. Dehghani et al. (2023) – NaViT
**Core Idea:**  
Use sequence packing during training to process images of arbitrary resolutions and aspect ratios (called Patch n' Pack). This reduces training cost if resolutions are randomly sampled during training while not degrading performance at the full resolution. Since the model performs well at all resolutions, smooth cost-performance tradeoff can be determined. It also generalizes with less cost to new tasks due to it not being fixed resolution.  
**Architecture Details:**  
They pack multiple images into a single sequence, and of course multiple sequences make a batch. Since it's impossible to pack all images in a batch such that all sequences are the same length, they must pad them all. They use a greedy packing algorithm, and find that typically less than 2% of tokens end up being padding tokens. 
They use masked attention to make sure image tokens only attend to other tokens of the same image. They also mask any pooling operations done on these tokens (ie for contrastive stuff).
They introduce factorized embeddings, where a separate embedding for each dimension (height and width) is learned separately and then added together. They explore various versions of this and determine that XYZ works best
**Strengths:**  
- Continually outperforms ViT at a fixed computational budget. Reason is that through sampling multple variable-resolution samples and token dropping they can pack more images into a batch than a vanilla ViT, resulting in more training examples seen with the same computational budget
**Limitations:**  
**Connection to §2.2:**  
Builds upon the concepts of ViTs by 

## 3. Lu & Elhamifar (2024) – FACT
**Core Idea:**  
**Temporal Modeling Mechanism:**  
**Efficiency Contributions:**  
**Limitations:**  
**Connection to §2.3:**  

## 4. Cheng et al. (2022) – Mask2Former
**Core Idea:**  
**Architecture:**  
**Innovation vs Prior Work:**  
**Limitations / Future Work:**  
**Connection to §2.4:**  
