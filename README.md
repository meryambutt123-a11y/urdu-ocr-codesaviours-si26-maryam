# Urdu OCR Project | Code Saviours SI-26 | Maryam Munawar

## Week 1 Overview
This repository contains the setup, research, and data collection for Week 1 of the Code Saviours ML / AI Internship Programme. The goal of this project is to build an Optical Character Recognition (OCR) model capable of reading Urdu text.

---

## Week 1, Monday Update: Project Research Summary

**1. What is OCR (Optical Character Recognition)?**
Optical Character Recognition (OCR) is a computer vision technology that converts images of typed, handwritten, or printed text into machine-readable digital text. Instead of simply viewing a document as a static picture, an OCR system acts like a human eye to recognize the unique shapes of individual letters and words. Once the text is successfully extracted, computers can process it just like a standard text file, allowing users to edit the content, search for keywords, or feed the data into other artificial intelligence models.

**2. Why is Urdu OCR harder than English OCR?**
Urdu OCR presents significant technical challenges primarily because it is written in the complex, cursive Nastaliq script where letters connect dynamically. Unlike English, which consists of distinct, isolated letters printed horizontally on a straight baseline, Urdu characters physically change their shape depending on whether they appear at the beginning, middle, or end of a word. Furthermore, Urdu text relies heavily on overlapping strokes and tiny dots (nuqtas) that drastically alter a word's meaning, making it incredibly difficult for standard algorithms to segment and recognize individual characters accurately.

**3. What are 2 real-world situations where Urdu OCR would be useful?**
* **Archiving and Digitization:** One major application of Urdu OCR is the mass digitization of historical archives and literature, making decades of printed books, newspapers, and poetry globally searchable and accessible for modern readers. 
* **Automated Data Entry:** A second vital use case is automating administrative workflows within legal and government sectors, allowing physical forms, national identity cards, and legal rulings to be instantly scanned into searchable digital databases. Implementing OCR in these situations drastically reduces the need for manual data entry, saving thousands of hours while preserving important physical documents from permanent loss.

---

## Week 1, Tuesday Update: Data Collection & Organization

Successfully collected a balanced dataset of 139 Urdu text images to train the model, treating the data like flashcards for the neural network. The images are systematically organized into the following required directory structure:

* `data/raw/newspaper` - Populated with 20 screenshots of headlines and article lines from Pakistani news websites.
* `data/raw/books` - Populated with 40 pristine, single-line printed sentences from Urdu literature.
* `data/raw/signboards` - Populated with 30 isolated real-world tokens and location names.
* `data/raw/synthetic` - Populated with 20 script-generated baseline images using the NotoNastaliqUrdu font.
* `data/raw/other` - Populated with 29 noisy digital/video text lines to act as a stress test for the model.

---

## Week 1, Wednesday Update: Data Engineering & PyTorch Pipeline

To prepare the raw images for machine learning, the dataset required strict standardization, ground truth labeling, and a mathematically robust loading pipeline.

**1. Persistent Cloud Architecture & Synthetic Generation**
Migrated the dataset from temporary computational environments to a permanent Google Drive directory structure. To complete the dataset, a Python script utilizing the `Pillow` (PIL) library was developed to dynamically generate 20 synthetic training images. These images were rendered using the `NotoNastaliqUrdu` TrueType font on standard 900x150 pixel canvases with correct right-to-left (RTL) anchoring, successfully populating the `data/raw/synthetic` directory.

**2. Automated CSV Pipeline & Ground Truth Annotation**
An automation script leveraging `pandas` and `os` was executed to scan all permanent image directories and map relative file paths to a master `labels.csv` file. Once generated, manual data annotation was completed to provide the ground truth for the model. The exact Urdu text for all 139 images across the five directories was entered and verified, resulting in a perfectly clean dataset with 0 missing labels.

**3. PyTorch Dataset Integration**
Transitioned the dataset into the machine learning phase by building a custom PyTorch `Dataset` class (`UrduOCRDataset`). A `torchvision.transforms` pipeline was implemented to standardize the visual data for neural network ingestion. The preprocessing steps include:
* **Resizing:** Uniformly reshaping all input images to `64x256` pixels.
* **Grayscale Conversion:** Reducing images to a single color channel to optimize OCR feature extraction.
* **Tensor Conversion & Normalization:** Converting raw PIL images into standardized mathematical PyTorch Tensors. 

The data pipeline is now fully integrated, allowing the dataset to be loaded directly into a neural network architecture for training.

---

## Week 2: Computer Vision Preprocessing & Baseline Evaluation

### Week 2, Monday Update: Phase Overview & Environment Setup
Raw image datasets contain inconsistencies in lighting, dimensions, and noise. The primary objective of this phase is to engineer an automated computer vision pipeline that standardizes the Urdu text images before feeding them into any deep learning architecture. 

Additionally, this phase establishes a baseline accuracy metric by evaluating an off-the-shelf OCR engine (Tesseract) against the complexities of cursive Nastaliq script.

**Tech Stack & Libraries**
* **OpenCV (`cv2`):** Core library for image transformations and matrix operations.
* **Pillow (`PIL`):** Utilized for high-level image file handling.
* **PyTesseract:** Python wrapper for the Tesseract OCR engine.

---

### Week 2, Tuesday Update: Preprocessing Pipeline Implementation
The core image processing pipeline was initially engineered in Python using OpenCV (`cv2`) to standardize input tensor dimensions, neutralize chromatic noise, and enhance text boundaries for subsequent sequence analysis.

```python
def preprocess_image(image_path, save_path):
    # Load the raw image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f'Could not load: {image_path}')
        return None
    
    # Step 1: Convert to grayscale (removes RGB color noise)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Resize to uniform dimensions (crucial for neural network inputs)
    resized = cv2.resize(gray, (512, 128))
    
    # Step 3: Non-Local Means Denoising (smooths out pixel artifacts)
    denoised = cv2.fastNlMeansDenoising(resized, h=10)
    
    # Step 4: Binarize (converts to pure black and white via absolute thresholding)
    _, binary = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)

    # Save the processed image into the output folder
    cv2.imwrite(save_path, binary)
    return binary
```

---

### Week 2, Wednesday Update: Automated Batch Processing & Algorithm Optimization

**Dataset Standardization**
The core OpenCV preprocessing pipeline was applied to the entire collected dataset. A batch processing script was engineered using Python's `glob` and `os` modules to recursively locate and transform all raw image assets.
* **Input:** 139 raw images of varying dimensions, lighting conditions, and formats across 5 distinct subdirectories.
* **Output:** 139 uniform (512x128), grayscale, denoised, and binarized image tensors successfully saved to the `data/processed/` directory.
* **Efficiency:** The automated loop ensures zero data loss and prepares the exact ground truth mapping required for the neural network training phase.

**Algorithm Optimization: Otsu's Binarization**
Initial batch processing revealed that a static binarization threshold (127) degraded the structural integrity of low-contrast source material (e.g., scanned literature). The preprocessing function was refactored to implement Otsu's Thresholding (`cv2.THRESH_OTSU`), which dynamically calculates the optimal contrast threshold per image based on its unique pixel histogram. Additionally, the non-local means denoising weight was reduced to preserve the delicate strokes and nuqtas characteristic of Nastaliq script.

---

### Week 2, Thursday Update: Baseline OCR Evaluation (Gap Analysis)

To establish a baseline accuracy metric and justify the requirement for a custom deep learning architecture, the preprocessed Nastaliq dataset was evaluated against Tesseract OCR (`tesseract-ocr-urd`), an industry-standard open-source OCR engine.

**Inference Results & Failure Modes**
A sample batch of normalized image tensors (512x128, binarized) was passed through the PyTesseract wrapper. The engine consistently produced catastrophic character error rates (CER) and completely failed to extract meaningful sequences. The failure modes were categorized as follows:
* **Segmentation Failure:** Tesseract's bounding-box character extraction algorithm assumes horizontal, non-overlapping glyphs. It failed to isolate Nastaliq's vertically stacked and diagonally cascading ligatures.
* **Contextual Misclassification:** The engine struggled to identify identical characters that dynamically changed shape based on their positional context (initial, medial, final).
* **Diacritic Orphaning:** Nuqtas (dots) were frequently disconnected from their base glyphs, resulting in persistent character hallucinations (e.g., misclassifying 'Te' as 'Be').

**Conclusion:** The baseline evaluation mathematically proved that traditional segmentation-based OCR systems are incompatible with cursive Nastaliq script. A sequence-to-sequence Deep Neural Network (CRNN/Attention-based) is strictly required to solve this problem without relying on character-level bounding boxes.

---

### Week 2, Friday Update: Proposed Custom Architecture Blueprint

Based on the catastrophic failure of standard segmentation-based OCR, the project will pivot to developing a custom end-to-end Deep Learning architecture capable of sequence-to-sequence translation without character-level bounding boxes. 

The proposed architecture for Week 3 is a **Convolutional Recurrent Neural Network (CRNN) utilizing CTC Loss**.

**Architectural Components:**
1. **Feature Extraction (CNN):** A Convolutional Neural Network (utilizing a ResNet backbone) will ingest the standardized 512x128 grayscale tensors to extract spatial feature maps, capturing the intricate strokes and nuqta placements of Nastaliq script.
2. **Sequence Modeling (Bi-LSTM):** A Bidirectional Long Short-Term Memory network will process the CNN feature maps as a sequential time-series. This allows the model to understand the contextual dependencies of connected ligatures from both left-to-right and right-to-left perspectives.
3. **Alignment & Decoding (CTC Loss):** Connectionist Temporal Classification (CTC) Loss will be implemented to calculate the error between the model's predicted sequence and the ground truth labels. CTC is critical for this project as it dynamically aligns the unsegmented input image features with the output text, entirely bypassing the need for manual character localization.
