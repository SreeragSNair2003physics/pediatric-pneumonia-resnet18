# Trained model checkpoint

The selected model is an ImageNet-pretrained ResNet-18 fine-tuned for `NORMAL` (index 0) and `PNEUMONIA` (index 1). Its checkpoint was saved as `resnet18_best.pth` inside the original `pneumonia_results.zip` generated on Kaggle.

The checkpoint is omitted from the small GitHub browser-upload bundle because it is approximately 45 MB. GitHub's browser upload limit is 25 MiB per file. The repository contains the full training notebook, results and configuration, so a new checkpoint can be reproduced by rerunning the training workflow.

The selected checkpoint corresponds to fine-tuning epoch 4 by validation ROC-AUC. Its decision threshold is `0.39297768473625183`, chosen with a validation pneumonia-recall target of 95%.

SHA-256 of the saved `resnet18_best.pth` file: `e97b860e92908f207c9ce684fd4378c3b45c4fff64bff41d8daf5e1d54c4a382`.

Do not load checkpoint files from untrusted sources; PyTorch serialization formats can carry security risks. The model is for research and educational use only, not clinical decisions.
