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

---

### Week 3, Monday Update:

**1. Data Collection & Synthetic Generation**
* **Handwritten (`data/raw/other/`):** Captured 15 clean, single-line images of handwritten Urdu text, focusing on relatable and humorous everyday sentences to introduce natural human stroke variance.
* **Real-World Captures:** Sourced an additional 36 images across three categories: 15 digital newspaper headlines, 10 printed book sequences, and 11 real-world physical signboards.
* **Synthetic Data (`data/raw/synthetic/`):** Engineered a Python generation script using the `Pillow`, `arabic-reshaper`, and `python-bidi` libraries to programmatically render 10 new right-to-left (RTL) baseline images on 900x150 pixel canvases.

#### Tuesday: Dataset Expansion & Preprocessing
* **Action:** Expanded the dataset from 139 to 200+ images to improve model robustness.
* **Focus:** Increased diversity by incorporating varied fonts (Nastaliq, Naskh), lighting conditions, and real-world noise.
* **Result:** Updated labels.csv to map all new image entries to their respective Urdu text labels.

#### Wednesday: Environment Troubleshooting & Architecture
* **Action:** Resolved persistent environment dependency conflicts in Google Colab related to Hugging Face transformers and tokenizer backends.
* **Focus:** Successfully integrated TrOCRProcessor and bypassed caching issues to ensure a stable, production-ready development environment.

#### Thursday: Designing the UrduOCRDataset Class
* **Action:** Architected a refined UrduOCRDataset class to handle end-to-end image-label processing.
* **Technical Implementation:**
    * Implemented `__getitem__` to dynamically process images via TrOCRProcessor.
    * Configured label encoding with padding to a maximum sequence length of 128.
    * Ensured compatibility with PyTorch DataLoader for efficient batch processing.

#### Friday: Dataset Verification & Validation
* **Action:** Conducted final validation tests on the pipeline to ensure data integrity.
* **Validation Metrics:**
    * **Shapes:** Verified pixel_values as [3, 384, 384] and labels as [128].
    * **Split:** Executed a 160/40 (80/20) train-test split to ensure statistical robustness.
* **Result:** The pipeline is now fully verified and prepared for model training in Week 4.

---

### 🛠 Tools & Libraries Used
* **ML Frameworks:** PyTorch, Hugging Face Transformers.
* **Computer Vision:** OpenCV, Pillow (PIL).
* **Data Handling:** Pandas, NumPy, OS.
* **Platforms:** Google Colab, GitHub.

---

### Week 4, Monday Update: Pipeline Debugging & Comprehensive Pipeline Execution

By resolving critical data-path bugs and successfully running the deep learning pipeline, the scheduled objectives for Monday, Tuesday, Wednesday, and Friday were completely fulfilled ahead of schedule in a single high-efficiency development session.

#### Monday & Tuesday: Dataset Pathing Optimization & DataLoader Setup
* **Action:** Resolved a critical `FileNotFoundError` caused by legacy path assumptions in the master `labels.csv` file and configured training parameters.
* **Technical Implementation:** 
    * Refactored the `UrduOCRDataset` class to completely bypass the missing `data/processed/` directory. 
    * Updated the code to map relative file paths directly from the project root (`PROJECT_ROOT`), allowing the pipeline to pull directly from the unedited `data/raw/` directories.
    * Initialized PyTorch `DataLoader` instances with a batch size of 4 to stream images to the GPU efficiently.
    * Instantiated the native `torch.optim.AdamW` optimizer with a fine-tuning learning rate of `5e-5`.
* **Result:** Successfully validated and loaded 199 high-quality raw Urdu samples, splitting them dynamically into an 80/20 ratio: **159 training samples** and **40 isolated testing samples**.

#### Wednesday: Baseline Training Loop Execution
* **Action:** Initialized and executed the core fine-tuning sequence for the sequence-to-sequence TrOCR architecture.
* **Technical Implementation:** Ran a native PyTorch training loop over 3 complete epochs using backpropagation to update the model weights against the complex structural layouts of raw Urdu text images.
* **Result:** The network successfully demonstrated mathematical convergence. The training loss decreased steadily across the epochs, dropping from an initial batch high of **17.2922** down to an average final epoch loss of **3.5272**.

#### Thursday: Preliminary Inference & Early Error Analysis
* **Action:** Switched the baseline model to evaluation mode (`model.eval()`) and generated text predictions using a beam search decoding configuration on unseen data from the test split.
* **Focus:** Conducted live inference testing to establish a performance baseline on raw data.
* **Result & Observation:** The initial 3-epoch baseline model generated partial Urdu ligatures intermixed with unrecognized padding symbols (replacement characters / ``). 
* **Conclusion:** While the steep reduction in loss proves the network architecture is successfully learning the overarching structural task, 3 epochs are mathematically insufficient for the model to lock in the complex character mappings of the Nastaliq script from scratch. This gap dictates the next phase of the project: extending the training phase for an additional 5-10 epochs and implementing advanced decoding strategies (e.g., `num_beams=4`) during inference generation.

#### Friday: Proactive Model Archiving
* **Action:** Secured the model's updated parameters immediately following the completion of the training loop to ensure session persistence.
* **Technical Implementation:** Executed a permanent save using `.save_pretrained()` to write both the structural weights and the specialized image tokenizers to Google Drive to prevent data loss upon session termination.
* **Result:** Saved the fine-tuned model weights and processor state securely to `/content/drive/MyDrive/Urdu_OCR_Project/models/trocr_urdu_raw_baseline`.
