# Urdu OCR Project | Code Saviours SI-26 | Maryam Munawar

## Week 1 Overview
[cite_start]This repository contains the setup, research, and data collection for Week 1 of the Code Saviours ML / AI Internship Programme[cite: 1]. [cite_start]The goal of this project is to build an Optical Character Recognition (OCR) model capable of reading Urdu text[cite: 1].

---

## Part 1: Project Research Summary
*(Completed - Day 1)*

**1. [cite_start]What is OCR (Optical Character Recognition)?** [cite: 1]
Optical Character Recognition (OCR) is a computer vision technology that converts images of typed, handwritten, or printed text into machine-readable digital text. Instead of simply viewing a document as a static picture, an OCR system acts like a human eye to recognize the unique shapes of individual letters and words. Once the text is successfully extracted, computers can process it just like a standard text file, allowing users to edit the content, search for keywords, or feed the data into other artificial intelligence models.

**2. [cite_start]Why is Urdu OCR harder than English OCR?** [cite: 1]
Urdu OCR presents significant technical challenges primarily because it is written in the complex, cursive Nastaliq script where letters connect dynamically. Unlike English, which consists of distinct, isolated letters printed horizontally on a straight baseline, Urdu characters physically change their shape depending on whether they appear at the beginning, middle, or end of a word. Furthermore, Urdu text relies heavily on overlapping strokes and tiny dots (nuqtas) that drastically alter a word's meaning, making it incredibly difficult for standard algorithms to segment and recognize individual characters accurately.

**3. [cite_start]What are 2 real-world situations where Urdu OCR would be useful?** [cite: 1]
* **Archiving and Digitization:** One major application of Urdu OCR is the mass digitization of historical archives and literature, making decades of printed books, newspapers, and poetry globally searchable and accessible for modern readers. 
* **Automated Data Entry:** A second vital use case is automating administrative workflows within legal and government sectors, allowing physical forms, national identity cards, and legal rulings to be instantly scanned into searchable digital databases. Implementing OCR in these situations drastically reduces the need for manual data entry, saving thousands of hours while preserving important physical documents from permanent loss.

---

## Part 2: Data Collection & Organization
*(Completed - Day 2)*

[cite_start]Successfully collected a balanced dataset of over 100 Urdu text images to train the model, treating the data like flashcards for the neural network[cite: 1]. [cite_start]The images are systematically organized into the following required directory structure[cite: 1]:

* `data/raw/newspaper` [cite: 1] - Populated with 20 screenshots of headlines and article lines from Pakistani news websites.
* [cite_start]`data/raw/books` [cite: 1] - Populated with 40 pristine, single-line printed sentences from Urdu literature.
* [cite_start]`data/raw/signboards` [cite: 1] - Populated with 30 isolated real-world tokens and location names.
* `data/raw/synthetic` [cite: 1] - Setup for script-generated baseline images.
* [cite_start]`data/raw/other` [cite: 1] - Populated with 30 noisy digital/video text lines to act as a stress test for the model.
