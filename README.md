# Robust CIFAR-10 Classification with Limited Training Data

## Overview

This project focuses on building a robust image classification model using only **500 training images from CIFAR-10**, with 50 images per class.

The main challenge is that the hidden test set contains corrupted images with rotations, brightness and contrast changes, blur, noise, and partial occlusion. A model trained only on clean images can overfit and perform poorly on these shifted images.

The objective is to improve generalization using data augmentation, regularization, and a lightweight CNN trained from scratch, without pretrained weights.

## Objectives

* Train a CNN using only 500 selected CIFAR-10 training images.
* Split the data into 400 training images and 100 validation images.
* Design a lightweight architecture with fewer than 1,000,000 trainable parameters.
* Include Inception-style modules with parallel convolution branches.
* Evaluate the effects of geometric and photometric augmentation.
* Analyze classification errors and robustness to distribution shifts.
* Compare performance across three random seeds.

## Dataset

The project uses three datasets:

| Dataset        | Purpose                                                         |
| -------------- | --------------------------------------------------------------- |
| `train_small`  | 500 clean training images, with 50 images per class             |
| Validation set | 100 images used for model selection and hyperparameter tuning   |
| `clean_test`   | 10,000 original CIFAR-10 test images, used for final reporting  |
| `shifted_test` | 10,000 corrupted CIFAR-10 test images, used for final reporting |

The shifted dataset contains geometric and visual corruptions, including rotations, brightness and contrast changes, blur, noise, and partial occlusion. An additional hidden corruption may also be present.

**Data leakage prevention:** Test sets are used only for final evaluation. Hyperparameters are selected using validation data.

## Model Architecture

The model is a lightweight CNN trained from scratch.

* Convolutional stem for initial feature extraction
* Two Inception-style modules
* Parallel 1×1, 3×3, 5×5 convolution and pooling branches
* 1×1 convolutions for channel reduction
* Batch Normalization after convolutional layers
* Global Average Pooling
* Dropout with probability 0.3
* Linear classification layer for 10 CIFAR-10 classes

**Trainable parameters:** 39,226

This is below the assignment limit of 1,000,000 trainable parameters.

## Data Augmentation

Augmentation is applied on the fly to training images only.

### Geometric Augmentation

* Random rotation up to ±30°
* Random shifts up to 4 pixels
* Random zoom
* Random horizontal flipping

### Photometric Augmentation

* Brightness, contrast, and saturation changes
* Random grayscale conversion
* Gaussian blur
* Random noise

### Additional Techniques

* Cutout / Random Erasing
* CutMix with probability 0.5
* Label smoothing with a factor of 0.1

These techniques aim to improve robustness to changes in orientation, image appearance, blur, noise, and occlusion.

Vertical flipping was not beneficial in the reported experiment, while rotations up to ±90° reduced accuracy. Excessive augmentation can make learning difficult when the training dataset and model are small.

## Training Configuration

| Setting                 | Value                               |
| ----------------------- | ----------------------------------- |
| Optimizer               | AdamW                               |
| Learning rate           | 0.01                                |
| Weight decay            | 0.01                                |
| Batch size              | 32                                  |
| Maximum epochs          | 60                                  |
| Learning-rate scheduler | Cosine schedule                     |
| Dropout                 | 0.3                                 |
| Label smoothing         | 0.1                                 |
| Model selection         | Best validation checkpoint          |
| Training approach       | From scratch, no pretrained weights |

### Random Seeds and Dataset Splits

The experiment uses three random seeds to report mean and standard deviation.

* **Evaluation seeds:** `0, 1, 2`
* **Hyperparameter tuning seeds:** `100, 101`
* **Dataset split seed:** `0`
* **Shifted-test corruption seed:** `2024`

The dataset split seed determines the selected 500 images and the 400/100 training-validation split. The shifted-test seed fixes the corruption pipeline.

Results are reported as **mean ± standard deviation over three evaluation seeds**, with accuracy expressed as a percentage.

## Results

### Final Test Performance

| Evaluation                 | Accuracy (mean ± std) |
| -------------------------- | --------------------: |
| Final model — clean test   |           30.5 ± 1.1% |
| Final model — shifted test |           20.0 ± 0.6% |
| Baseline — clean test      |           42.4 ± 1.3% |
| Baseline — shifted test    |           17.2 ± 0.6% |

The final model improves shifted-test accuracy by **2.7 percentage points** compared with the plain baseline.

The lower accuracy on shifted images demonstrates the difficulty of generalizing from a small clean training set to corrupted images.

### Ablation Study

| Configuration                    | Validation accuracy (clean) | Validation accuracy (shifted) | Clean test accuracy | Shifted test accuracy |
| -------------------------------- | --------------------------: | ----------------------------: | ------------------: | --------------------: |
| Plain baseline                   |                 48.3 ± 0.5% |                   20.3 ± 0.9% |         42.4 ± 1.3% |           17.2 ± 0.6% |
| No augmentation, regularized     |                 44.3 ± 0.5% |                   17.7 ± 1.7% |         40.5 ± 0.6% |           17.2 ± 0.4% |
| Geometric only                   |                 37.3 ± 1.2% |                   22.3 ± 1.9% |         36.1 ± 2.9% |           18.7 ± 1.0% |
| Photometric only                 |                 44.3 ± 1.7% |                   19.7 ± 1.7% |         40.4 ± 1.4% |           18.6 ± 0.2% |
| Geometric + photometric          |                 36.7 ± 1.2% |                   24.3 ± 0.5% |         36.1 ± 2.5% |           20.8 ± 1.5% |
| Geometric + photometric + Cutout |                 36.0 ± 1.4% |                   25.0 ± 1.6% |         33.2 ± 1.9% |           21.2 ± 0.8% |
| Final: CutMix + label smoothing  |                 35.0 ± 4.5% |                   27.0 ± 2.2% |         30.5 ± 1.1% |           20.0 ± 0.6% |

*All results are reported as mean ± standard deviation over three seeds.*

### Main Finding

Combining geometric and photometric augmentation produced the largest clear improvement over the baseline: shifted-test accuracy increased from 17.2% to 20.8%, a gain of 3.6 percentage points.

Geometric augmentation helps with rotations and translations, while photometric augmentation improves robustness to brightness, contrast, and other appearance changes. Cutout provided a small additional improvement in the reported shifted-test results. Differences between later configurations should be interpreted cautiously because of seed variability.

## Error Analysis

The reported model performed best on ship, truck, and automobile classes, while cat and bird had very low classification accuracy.

Frequent confusions included:

* Airplane predicted as ship
* Automobile predicted as truck
* Deer predicted as frog

Occlusion can hide important object features, while noise and brightness changes can make different classes look similar. The RGB channel-swap experiment also showed limited generalization to the unlisted corruption.

## Test-Time Augmentation

Test-time augmentation was evaluated by averaging predictions across eight augmented views.

The reported change in shifted-test accuracy was −0.13 percentage points, with approximately eight times the inference cost. Therefore, it did not provide a meaningful benefit in this experiment.

## Limitations

* Only 400 images are used for model training.
* The small model has limited capacity to learn robust features.
* Aggressive augmentation can cause underfitting.
* Accuracy remains low on severely corrupted images.
* The shifted dataset was constructed from the assignment description, so the reported results may differ from those on the official hidden test set.

## Technologies Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* Seaborn

## How to Run

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
   ```

2. Install the required libraries:

   ```bash
   python -m pip install torch torchvision numpy pandas matplotlib seaborn
   ```

3. Place the provided datasets in the expected locations and ensure the dataset-loading paths match your code.

4. Run the training script:

   ```bash
   python your_script.py
   ```

   Replace `your_script.py` with the actual name of your Python file.

**Note:** The execution command and dataset paths may need adjustment to match the files in your repository.

## Conclusion

This project demonstrates how augmentation and regularization can improve the robustness of a CNN trained with very limited data. Combining geometric and photometric augmentation produced the strongest clear improvement over the baseline, although the final shifted-test accuracy remained modest. The experiments also show the importance of validation-based model selection, fixed random seeds, and careful error analysis when evaluating robustness under distribution shifts.
