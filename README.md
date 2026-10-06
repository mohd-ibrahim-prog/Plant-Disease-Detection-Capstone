# AI-Powered Plant Disease Detection & Analysis System

An AI/ML-based plant disease detection system that uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify plant leaf images into 38 supported plant health and disease classes.

The project progresses from dataset preparation and machine learning fundamentals to deep learning, model evaluation, per-class analysis, image validation, uncertainty handling, and a browser-based web application.

The final goal of the project is to provide a practical and user-friendly plant image analysis system that does not blindly return a disease prediction for every uploaded image. Instead, the application performs multiple validation and confidence checks before displaying a prediction.

---

# 1. Project Overview

Plant disease identification is an important task in agriculture because early identification of diseases can help farmers and agricultural users take appropriate action.

Traditional identification usually requires manual inspection or expert knowledge. Computer vision and deep learning can assist this process by learning visual patterns from plant leaf images.

This project develops an AI-powered plant disease classification system using a CNN trained on the PlantVillage dataset.

However, a major practical problem with a normal image classifier is that it may produce a prediction even when the uploaded image is:

- Not a plant
- A car or other object
- Too blurry
- Too dark or too bright
- Outside the training domain
- Ambiguous
- Visually different from the training dataset

Therefore, the project goes beyond simply calling the CNN and displaying its highest probability.

The application introduces an image validation and prediction-decision pipeline designed to **withhold unreliable predictions instead of presenting every prediction as a confirmed diagnosis**.

---

# 2. Problem Statement

A standard image classification model can return a high-confidence class even when the uploaded image does not belong to the type of images used during training.

For example, a user may upload:

- A car image
- A laptop image
- A person
- A random outdoor photograph
- A blurry image
- A very dark image
- An unrelated object

A conventional softmax classifier may still select one of its supported plant disease classes.

This creates a serious usability problem because a high softmax probability does not necessarily mean that the model actually understands the uploaded image.

Therefore, this project focuses on building a safer prediction workflow that checks the uploaded image before allowing the CNN result to be displayed.

---

# 3. Project Objectives

The major objectives of the project are:

1. Build an image classification model for plant disease detection.

2. Train the model using the PlantVillage dataset.

3. Classify images into 38 supported plant health/disease classes.

4. Evaluate the trained model using a held-out test dataset.

5. Analyze performance separately for individual classes.

6. Build a browser-based interface for image upload and prediction.

7. Validate uploaded images before prediction.

8. Detect images that are likely not plant/leaf images.

9. Detect poor-quality images such as blurry, extremely dark, or extremely bright images.

10. Use confidence, prediction margin, entropy, and consistency checks before accepting a prediction.

11. Avoid displaying a disease prediction when the system does not have sufficient evidence.

12. Provide a clear and user-friendly interface for the final application.

---

# 4. Project Development Journey

The project was developed progressively through the AI/ML Foundation learning process.

| Stage | Main Focus | Result |
|------|-------------|--------|
| Data Preparation | Dataset loading and cleaning | Clean ML-ready data |
| Machine Learning | Linear Regression | Regression model and evaluation |
| NLP | TF-IDF + Naive Bayes | SMS spam classifier |
| Deep Learning | CNN | Initial plant disease classifier |
| Model Training | Training + validation | Improved training workflow |
| Model Improvement | Test split + augmentation | Evaluated CNN |
| Model Analysis | Per-class evaluation | Detailed class-level analysis |
| Application | Flask web application | Browser-based plant analysis system |
| Capstone | Scope and prototype planning | Project refinement |
| Future | Deployment and final demo | Planned final delivery |

The earlier foundation work established the technical concepts required to build the final project.

---

# 5. Dataset

## PlantVillage Dataset

The project uses the PlantVillage dataset for plant disease image classification.

The dataset contains plant leaf images organized into different plant and disease/health categories.

The current model supports:

**38 classes**

The dataset is used for training, validation, testing, and evaluation.

The complete dataset is not stored inside this repository because of its size.

---

# 6. Supported Classes

The trained model currently supports the following 38 classes:

1. Apple — Apple Scab
2. Apple — Black Rot
3. Apple — Cedar Apple Rust
4. Apple — Healthy
5. Blueberry — Healthy
6. Cherry — Powdery Mildew
7. Cherry — Healthy
8. Corn — Cercospora Leaf Spot / Gray Leaf Spot
9. Corn — Common Rust
10. Corn — Northern Leaf Blight
11. Corn — Healthy
12. Grape — Black Rot
13. Grape — Esca / Black Measles
14. Grape — Leaf Blight
15. Grape — Healthy
16. Orange — Haunglongbing / Citrus Greening
17. Peach — Bacterial Spot
18. Peach — Healthy
19. Pepper Bell — Bacterial Spot
20. Pepper Bell — Healthy
21. Potato — Early Blight
22. Potato — Late Blight
23. Potato — Healthy
24. Raspberry — Healthy
25. Soybean — Healthy
26. Squash — Powdery Mildew
27. Strawberry — Leaf Scorch
28. Strawberry — Healthy
29. Tomato — Bacterial Spot
30. Tomato — Early Blight
31. Tomato — Late Blight
32. Tomato — Leaf Mold
33. Tomato — Septoria Leaf Spot
34. Tomato — Spider Mites
35. Tomato — Target Spot
36. Tomato — Tomato Yellow Leaf Curl Virus
37. Tomato — Tomato Mosaic Virus
38. Tomato — Healthy

The model should only be considered reliable within this supported class/domain.

---

# 7. Machine Learning & Deep Learning Workflow

The overall development workflow is:

```text
Dataset
   ↓
Data Preparation
   ↓
Image Preprocessing
   ↓
Train / Validation / Test Split
   ↓
CNN Training
   ↓
Data Augmentation
   ↓
Early Stopping
   ↓
Model Checkpointing
   ↓
Held-Out Test Evaluation
   ↓
Per-Class Evaluation
   ↓
Web Application
   ↓
Image Validation
   ↓
Plant / Non-Plant Check
   ↓
CNN Prediction
   ↓
Confidence & Uncertainty Checks
   ↓
Final Decision
   ↓
Prediction OR Prediction Withheld
