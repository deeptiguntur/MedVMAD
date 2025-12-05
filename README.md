# MedVMAD

MedVMAD:Medical imaging Anomaly Detection using Visual Language Models


MedVMAD is a framework that uses visual-language models such as CLIP for **anomaly detection** and **localization** in medical images (e.g., brain MRI, Liver-CT). The system aligns image features with clinically relevant text embeddings to identify abnormal regions without requiring extensive labeled datasets. [Paper](Research_paper.pdf). 

## Method Overview

Our approach adapts CLIP for medical anomaly detection by integrating learnable text prompts and learnable image feature token embeddings. The method consists of the following key steps:

1. **Prompt Engineering and Adaptation:**  
    We initialize CLIP with domain-specific prompts describing "normal" and "abnormal" medical images. These prompts are made learnable, allowing  the model to optimize their textual representations during training and improve its ability to distinguish between normal and abnormal cases.

2. **Visual Feature Adaptation:**  
    The image encoder’s output tokens are refined using learnable adapters. These adapters adjust the visual features to emphasize subtle anomalies and pixel-level differences, enabling more accurate localization of abnormal regions.

3. **Loss Function:**  
    The model is optimized using a combined loss that incorporates both text-level and pixel-level embedding losses. The text loss guides the model in classifying images, while the pixel-level loss supports accurate localization of anomalies. Together, they encourage alignment between adapted image features and optimized text prompts, enabling the model to differentiate normal and abnormal cases based on similarity scores.

4. **Localization and Zero-Shot Capability:**  
    By leveraging CLIP’s ability to generalize, our framework supports zero-shot anomaly detection and localization. The model can identify and localize anomalies in unseen images by comparing them to the learned textual descriptions.

5. **Evaluation:**  
    The method is evaluated on the BMAD (BRaTS2021) dataset, focusing on brain MRI scans. Performance is measured in terms of anomaly classification accuracy and localization precision.


Quick start

1. Install dependencies

Ensure you have a Python 3.8+ environment and install required packages:

```powershell
pip install -r requirements.txt
```

2. Prepare data

- Unzip the provided `data.zip` or place your medical image datasets under the `data/` directory. Example dataset generators are in `generate_dataset/`.
- Dataset loading logic is implemented in `dataset.py`.

3. Use a checkpoint (optional)

Pretrained checkpoints (training snapshots) are stored under the `checkpoint/` and `final_checkpoints/` directories.

4. Train

Run the training script to start training a model (see `train.py` and `train.sh` for examples). Training will save checkpoints to `checkpoint/`.

5. Test / Inference

Use `test.py` or the example scripts `test_one_example.py` / `test_one_example.sh` to run evaluation on a test split or a single image. Visualized outputs are saved under the `output/` and `results/` directories.


Configuration and customization

- Edit the training and testing arguments in `train.py` and `test.py` to change hyperparameters, dataset paths, and checkpoint locations.
- Prompt sets and ensembles can be modified in `prompt_ensemble.py` to experiment with different textual descriptions.
- The `CLIP/MedVMAD.py` module contains the integration logic between the visual encoder and prompt-based scoring.

