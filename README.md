AI-Powered Plant Disease Detection & Analysis System

An AI/ML-based plant leaf image analysis system that combines a CNN classifier with an image-validation and uncertainty-aware decision layer, so it can either ACCEPT a prediction or WITHHOLD it and explain why.

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image
Table of Contents
Status Legend
Project Overview
Problem Statement
Project Objectives
Core Project Idea
Why This Project Is Different
Dataset
Supported Classes
System Workflow
CNN Model
Image Preprocessing
Training / Validation / Testing
Data Augmentation
Early Stopping and Model Checkpointing
Model Evaluation
Per-Class Evaluation
Web Application
Application Architecture
Backend Components
Configuration
File Validation
Image Quality Analysis
Blur Detection
Plant / Non-Plant Screening
CNN Prediction
Confidence Analysis
Prediction Margin
Entropy / Uncertainty
Prediction Consistency
OOD / Domain Handling
Final Decision Engine
Prediction Evidence
Top-3 Predictions
User Interface
Project Structure
Capstone Scope
Working Prototype
Deployment
Final Demonstration
Testing Strategy
Limitations
Responsible Use
Technical Skills
Future Scope
Project Roadmap
Final System Architecture
Final Project Vision
Current Status
Conclusion
Disclaimer
Status Legend

To keep this document honest about what exists today and what does not, statements throughout this README are tagged where it matters:

Tag	Meaning
CURRENT	Developed as part of the existing AI/ML work and carried into this Capstone repository as the project basis
PLANNED	Targeted for the Capstone phases (prototype integration, deployment, final demonstration)
FUTURE SCOPE	Ideas for later improvement. These are not implemented and not promised for the Capstone

Note: The project is currently in the Capstone Scope Finalization phase. The working prototype is the next milestone, and deployment is planned but not yet completed.

1. Project Overview
What the project is

AI-Powered Plant Disease Detection & Analysis System is an AI/ML-based system that uses a Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify plant leaf images into 38 supported plant health and disease classes.

The project goes beyond a basic image classifier. Between the user's uploaded image and the final answer, it adds a validation and decision-making pipeline. That pipeline checks whether the image is valid, whether its quality is good enough, whether it looks plant-like, and whether the CNN's output is trustworthy enough to show.

Why plant disease detection matters

Plant diseases can reduce crop health and yield, and early recognition of visible symptoms helps people take timely action. Many symptoms appear on leaves as spots, discoloration, mold, rust-like patterns, or curling. Recognizing these visual patterns by eye needs experience, and not everyone who grows plants has access to an expert at the moment they need one.

Why computer vision and deep learning can help

CNNs can learn visual patterns such as edges, textures, color distributions, and lesion shapes directly from labeled images. A well-trained CNN can pick up on subtle visual cues that are hard to describe with hand-written rules. This makes deep learning a natural fit for leaf-image classification.

Why a normal CNN classifier is not enough

A conventional CNN classifier is a closed-set classifier. It is trained on a fixed list of classes (here, 38), and its final softmax layer always produces a probability distribution over exactly those 38 classes. The probabilities always sum to 1, so something always ends up on top.

This means a standard classifier may still return one of its known classes even when the uploaded image is:

A car
A laptop
A person
A building
A random object
A very blurry image
A very dark image
A very bright / overexposed image
A plant species that is not supported
A disease that is not supported
An image significantly different from the training distribution

Worse, the model may present such an answer with a high softmax probability. A high softmax value does not automatically mean the prediction is correct. It only says that, among the 38 known classes, one class received the largest share of the probability mass. It does not say that the image belongs to any of those classes in the first place.

What this system does differently

This project is built around one core philosophy:

"Do not force the model to make a prediction when the available evidence is not reliable enough."

Instead of the simple flow Image → CNN → Disease, this system follows:

Uploaded Image
→ File Validation
→ Image Quality Analysis
→ Plant / Non-Plant Screening
→ Image Preprocessing
→ CNN Inference
→ Confidence Analysis
→ Prediction Margin Analysis
→ Entropy / Uncertainty Analysis
→ Prediction Consistency Analysis
→ Domain / OOD-related Checks
→ Final Decision
→ Either ACCEPT prediction OR WITHHOLD prediction

When the evidence is insufficient, the system withholds the prediction and explains why, rather than presenting an unjustified disease label.

What the final application is intended to provide

The intended final deliverable is a browser-based web application where a user can:

Upload a plant leaf image (by file picker or drag and drop)
Preview it
Run analysis
Receive either:
an accepted prediction with class name, confidence, top-3 ranked outputs, and supporting evidence, or
a withheld result with a clear reason and guidance on what to do next
2. Problem Statement

A CNN trained on a fixed set of PlantVillage classes expects inputs from a similar visual domain. Images in PlantVillage are typically leaf photographs captured under relatively controlled conditions. However, users of a real application can upload arbitrary images: photographs with cluttered backgrounds, different cameras, unusual lighting, multiple leaves, unsupported plants, or no plant at all.

A standard classifier may still assign one of its 38 classes to an unrelated or unreliable image. This creates a reliability problem: the system produces confident-looking output that is not backed by real evidence.

Key concepts
Concept	Meaning in this project
Classification	Assigning an input image to one of the known classes the model was trained on
Confidence	The probability value the model assigns to its top predicted class (softmax output). It describes the model's internal score distribution, not the real-world probability of being correct
Uncertainty	How spread out or ambiguous the model's output is. For example, several classes with similar probabilities indicate higher uncertainty
Supported domain	The set of inputs the model was designed and trained for: leaf images of the supported plants and conditions in the 38-class PlantVillage setup
Unsupported input	Anything outside that domain: non-plant images, other plant species, other diseases, very poor quality images, or images that differ greatly from the training distribution
Why a validation and decision layer is added

Because the CNN cannot say "this is not something I was trained on", the project wraps it in additional checks. These checks act as additional signals that attempt to identify unreliable situations and help reduce the chance of showing a misleading prediction. When the signals indicate insufficient evidence, the prediction may be withheld.

Important: This project does not claim to completely solve out-of-distribution (OOD) detection. The validation and decision layer is heuristic. It is meant to make the system more cautious and transparent, not perfect.

3. Project Objectives
Plant disease image classification: build a system that classifies plant leaf images into plant health and disease classes.
38 supported classes: support the 38 PlantVillage plant–condition classes listed in this README.
PlantVillage dataset: use the PlantVillage dataset as the basis for training and evaluation.
CNN development: design and train a Convolutional Neural Network for multi-class image classification.
Training / validation / testing: organize the data into training, validation, and test sets (70% / 15% / 15%) and use each appropriately.
Model evaluation: evaluate the model with accuracy, precision, recall, F1-score, confusion matrix, and classification report.
Per-class evaluation: analyze performance for each class to identify strong and weak classes and common confusions.
Browser-based application: provide a web interface that runs in desktop and mobile browsers.
Image upload: allow users to upload leaf images through the interface.
File validation: verify file existence, extension, decodability, dimensions, and size before analysis.
Image quality analysis: evaluate dimensions, sharpness (blur), brightness, and contrast.
Plant / non-plant screening: apply a lightweight vegetation/color-based heuristic to check for plant-like visual characteristics.
Confidence thresholding: use the CNN's top-1 confidence as one decision signal.
Prediction margin: compare top-1 and top-2 probabilities to detect ambiguity.
Entropy-based uncertainty: measure how spread out the probability distribution is.
Prediction consistency: check whether the prediction remains stable under small image transformations.
Prediction withholding: allow the system to decline to show a prediction and give a reason.
Top-3 predictions: display the three highest-ranked model outputs for context.
Evidence display: show the supporting signals behind an accepted prediction for transparency.
User-friendly UI: provide a clear, responsive interface with loading states, results, retry, and limitations information.
Working prototype: integrate all components into a functioning end-to-end application. (PLANNED)
Deployment: deploy the application to a public web URL, targeting Render. (PLANNED)
Final documentation: deliver complete, accurate project documentation.
Final demonstration: demonstrate accepted and withheld scenarios end to end. (PLANNED)
4. Core Project Idea

The project's central idea is a pipeline that treats the CNN as one component within a larger evidence-based decision process.

                      User Image
                          │
                          ▼
                   File Validation
                          │
                          ▼
                Image Quality Analysis
                          │
                          ▼
              Plant / Non-Plant Screening
                          │
                          ▼
                 Image Preprocessing
                          │
                          ▼
                    CNN Prediction
                          │
                          ▼
                  Confidence Analysis
                          │
                          ▼
                   Prediction Margin
                          │
                          ▼
                Entropy / Uncertainty
                          │
                          ▼
               Prediction Consistency
                          │
                          ▼
           Domain / OOD-related Checks
                          │
                          ▼
                    Final Decision
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
     Prediction Accepted       Prediction Withheld
             │                         │
             ▼                         ▼
     Class + Evidence            Explanation
Stage-by-stage explanation
Stage	Purpose
File Validation	Confirms the upload is a real, decodable image in a supported format and within allowed limits. Rejects broken or unsupported files early.
Image Quality Analysis	Measures dimensions, sharpness, brightness, and contrast. Images that are too small, too blurry, too dark, too bright, or too flat may produce unreliable predictions.
Plant / Non-Plant Screening	Uses a lightweight color-based vegetation heuristic to check whether the image contains plant-like visual characteristics. Helps screen out obviously unrelated images.
Image Preprocessing	Converts the validated image into the exact numerical format the CNN expects (RGB, 128 × 128, float32, normalized).
CNN Prediction	Produces a probability distribution over the 38 supported classes.
Confidence Analysis	Examines the top-1 probability as one signal of how strongly the model favors its top class.
Prediction Margin	Compares top-1 and top-2 probabilities. A small gap suggests ambiguity between two classes.
Entropy / Uncertainty	Measures how concentrated or spread out the probability distribution is.
Prediction Consistency	Re-evaluates the image under small transformations to see if the prediction stays stable.
Domain / OOD-related Checks	Combines the available signals (confidence, margin, entropy, consistency, quality, plant-likelihood) as heuristic evidence about whether the image belongs to the supported domain.
Final Decision	Applies the decision logic to either accept or withhold the prediction.
Accepted / Withheld output	Accepted: class, confidence, top-3, evidence. Withheld: the reason and guidance.
5. Why This Project Is Different
A basic classifier
Image
  ↓
CNN
  ↓
Highest Probability
  ↓
Display Result

A basic classifier always produces an answer. It cannot say "I don't have enough evidence."

This system
Image
  ↓
Validate
  ↓
Check Quality
  ↓
Check Plant-Likelihood
  ↓
CNN
  ↓
Analyze Confidence
  ↓
Analyze Margin
  ↓
Analyze Entropy
  ↓
Check Consistency
  ↓
Consider Domain
  ↓
Final Decision
  ↓
Accept OR Withhold
Side-by-side comparison
Aspect	Basic CNN Classifier	This System
Behavior on non-plant images	May still output one of 38 classes	Attempts to identify and withhold
Behavior on blurry/dark/bright images	Outputs a label regardless	Checks quality and may withhold
Use of confidence	Often only shows top-1 label	Uses confidence as one of several signals
Ambiguity between two classes	Hidden	Analyzed using top-1 vs top-2 margin
Spread-out probabilities	Ignored	Analyzed using entropy
Stability under minor changes	Not checked	Checked via prediction consistency
Explanation	Typically none	Evidence shown for accepted results, reasons for withheld results
Possible outcomes	One (a class label)	Two major outcomes: Accept or Withhold
Why the second approach is more practical and cautious

In practice, users do not always upload clean, in-domain images. A system that can recognize when it should not answer is more cautious and more transparent than one that always answers. It gives users a clearer sense of when a result is supported by evidence.

Scope note: This is an academic project. It is not medically or agriculturally certified and is not claimed to be production-grade.

6. Dataset
PlantVillage

The project uses the PlantVillage dataset, a widely used image dataset for plant disease classification. It contains labeled leaf images covering multiple plant species and their healthy and diseased conditions.

How it is used in this project
Purpose	Description
Image-based plant disease classification	Leaf images are used as inputs and plant/condition labels as targets
38 supported classes	The model predicts one of 38 plant–condition classes
Training	Learning the visual patterns of each class
Validation	Monitoring performance during training, early stopping, and checkpoint selection
Testing	Final evaluation on held-out data
Per-class analysis	Examining performance class by class
Dataset availability

The complete dataset is NOT committed to this GitHub repository because of its size. To retrain the model, the dataset must be obtained separately and placed locally according to the project's data-loading setup. A dataset URL is intentionally not listed here because none has been provided for this documentation.

7. Supported Classes

The system supports exactly the following 38 classes:

#	Plant	Condition
1	Apple	Apple Scab
2	Apple	Black Rot
3	Apple	Cedar Apple Rust
4	Apple	Healthy
5	Blueberry	Healthy
6	Cherry	Powdery Mildew
7	Cherry	Healthy
8	Corn	Cercospora Leaf Spot / Gray Leaf Spot
9	Corn	Common Rust
10	Corn	Northern Leaf Blight
11	Corn	Healthy
12	Grape	Black Rot
13	Grape	Esca / Black Measles
14	Grape	Leaf Blight
15	Grape	Healthy
16	Orange	Haunglongbing / Citrus Greening
17	Peach	Bacterial Spot
18	Peach	Healthy
19	Pepper Bell	Bacterial Spot
20	Pepper Bell	Healthy
21	Potato	Early Blight
22	Potato	Late Blight
23	Potato	Healthy
24	Raspberry	Healthy
25	Soybean	Healthy
26	Squash	Powdery Mildew
27	Strawberry	Leaf Scorch
28	Strawberry	Healthy
29	Tomato	Bacterial Spot
30	Tomato	Early Blight
31	Tomato	Late Blight
32	Tomato	Leaf Mold
33	Tomato	Septoria Leaf Spot
34	Tomato	Spider Mites
35	Tomato	Target Spot
36	Tomato	Tomato Yellow Leaf Curl Virus
37	Tomato	Tomato Mosaic Virus
38	Tomato	Healthy

Any plant species or disease not in this list is outside the supported domain.

8. System Workflow

This diagram shows the complete ML + application workflow, from the dataset through the final user-facing result.

                     ┌───────────────────────────┐
                     │   PlantVillage Dataset    │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │    Data Preparation       │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │      Preprocessing        │
                     │ (RGB, 128×128, normalize) │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │ Train / Validation / Test │
                     │     70%  /  15% / 15%     │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │        CNN Model          │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │         Training          │
                     │  + Data Augmentation      │
                     │  + Early Stopping         │
                     │  + Model Checkpointing    │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │     Test Evaluation       │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │   Per-Class Analysis      │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │      Selected Model       │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │     Web Application       │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │   Validation Pipeline     │
                     │ (file, quality, plant)    │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │      Decision Layer       │
                     │ (confidence, margin,      │
                     │  entropy, consistency)    │
                     └─────────────┬─────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │       Final Result        │
                     │  Accepted  /  Withheld    │
                     └───────────────────────────┘

The workflow has two halves:

Model development (offline): dataset → training → evaluation → selected model.
Application (online, per request): upload → validation → CNN → decision layer → result.
9. CNN Model
What a CNN is

A Convolutional Neural Network (CNN) is a type of neural network designed for grid-like data such as images. Instead of treating every pixel independently, a CNN learns small filters that slide across the image and respond to local visual patterns.

Why CNNs suit image classification
Images have spatial structure: nearby pixels are related.
The same visual pattern (an edge, a spot, a texture) can appear in different places. Convolution shares filters across positions.
Early layers tend to learn simple features such as edges and color transitions; deeper layers can combine them into more complex patterns, such as lesion-like textures.
Concepts used in this project's model
Concept	Role
Convolution	Filters extract local visual features such as edges, textures, and color patterns from the image
Pooling	Reduces spatial size while retaining the most important responses, lowering computation and adding some tolerance to small shifts
Dropout	Randomly disables a fraction of units during training to reduce overfitting
Dense (fully connected) layers	Combine the extracted features to make a classification decision
Softmax	Final activation that converts raw scores into a probability distribution across the classes
38-class output	The output layer produces one probability for each of the 38 supported classes

Architecture note: This README describes the model at a conceptual level. Exact layer counts, filter sizes, and parameter counts are intentionally not stated here, because they are not specified in this documentation. They can be added from the actual training code or model summary.

10. Image Preprocessing

Before an image reaches the CNN, it goes through the following preprocessing steps:

Open Image
   ↓
Validate Image
   ↓
Convert to RGB
   ↓
Resize to 128 × 128
   ↓
Convert to float32
   ↓
Normalize Pixel Values
   ↓
Send to CNN
Step	Description
Open image	The uploaded file is opened and decoded
Validate image	Confirms the file is a valid image that can be read
RGB conversion	Ensures three color channels (for example, converting images that are grayscale or have an alpha channel)
Resize to 128 × 128	Matches the CNN's expected input size
Convert to float32	Converts pixel data to the numeric type used by the model
Normalize pixel values	Scales pixel values into the range/representation the model expects
Send to CNN	The preprocessed array is passed to the model for inference

Consistency requirement: Inference-time preprocessing must remain consistent with the model's expected input representation, meaning the same size, channel order, data type, and normalization used when the model was trained. Any mismatch between training-time and inference-time preprocessing can silently degrade predictions.

11. Training / Validation / Testing

The dataset is divided into three subsets:

Split	Share	Role
Training set	70%	Used to update the model's weights. The model learns visual patterns from these images.
Validation set	15%	Used during training to monitor generalization, support early stopping, and select the best checkpoint.
Test set	15%	Held out until the end, then used to estimate performance on data the model has not been trained or tuned on.
Why the test data must not be used for training

If test images influence training or model selection, the test score no longer measures how well the model handles unseen data. It becomes an inflated estimate. Keeping the test set separate gives a more honest evaluation of generalization within the PlantVillage setup.

12. Data Augmentation

Data augmentation creates modified versions of training images during training, so the model sees more variation than the raw dataset alone provides.

The augmentations used are:

Augmentation	Purpose
Random horizontal flipping	Teaches the model that left–right orientation of a leaf should not change its class
Random rotation	Adds tolerance to different leaf orientations
Random zoom	Adds tolerance to differences in how large the leaf appears in the frame
Why augmentation is used
Reduces overfitting by exposing the model to more varied views of the same underlying patterns.
Encourages the model to focus on disease-related features rather than exact positions or orientations.
Helps the model handle small variations in how images are framed.

No other augmentation techniques are claimed for this project.

13. Early Stopping and Model Checkpointing
Early stopping

Training for too long can make a model memorize the training data (overfitting). Early stopping monitors validation performance and halts training when it stops improving for a set number of epochs. It can also restore the best weights observed, so the final model corresponds to the best validation point rather than the last epoch.

Model checkpointing

Model checkpointing saves the model during training whenever validation performance improves. This ensures the best-performing model is preserved even if later epochs perform worse.

Together
Mechanism	Benefit
Early stopping	Avoids unnecessary training and reduces overfitting
Restoring best weights	Returns the best validation state, not just the final one
Model checkpointing	Preserves the best model on disk for use in the application
Selecting best-performing model	Provides the final model loaded by the web application
14. Model Evaluation
Metrics used
Metric	What it tells us
Accuracy	Overall fraction of test images classified correctly
Precision	Of the images predicted as a class, how many truly belong to it
Recall	Of the images that truly belong to a class, how many the model found
F1-score	Harmonic mean of precision and recall, balancing both
Confusion Matrix	Shows which classes are confused with which
Classification Report	Per-class precision, recall, F1-score, and support
Recorded result

The model developed during the foundation phase of this project (referred to in project notes as the "Week 6 model") achieved approximately 92.89% test accuracy on the held-out PlantVillage test setup used during development.

Important: this is not a real-world accuracy guarantee

92.89% is NOT a guarantee of real-world accuracy. It reflects performance on held-out PlantVillage data. Real-world photographs may differ because of:

Lighting: shadows, glare, low light, direct sunlight
Background: soil, hands, clutter, other plants
Camera quality: resolution, lens, noise, compression
Occlusion: leaf partially hidden or overlapped
Multiple leaves: several leaves or whole plants in one frame
Different varieties: plant varieties not well represented in training
Different disease stages: early, mixed, or advanced symptoms
Unseen domains: plants or conditions not in the 38 classes

This gap is exactly why the project adds a validation and decision layer.

15. Per-Class Evaluation
Why overall accuracy is not enough

A single accuracy number can hide important differences. A model can score well overall while performing poorly on certain classes, particularly classes that are visually similar to others or have fewer examples.

What is examined
Item	Purpose
Per-class accuracy / performance	How well each individual class is recognized
Precision	How trustworthy predictions of each class are
Recall	How completely each class is detected
F1-score	Balanced per-class summary
Support	Number of test samples per class, which helps interpret the reliability of per-class numbers
Strong classes	Classes the model recognizes reliably
Weak classes	Classes with lower precision/recall that need attention
Confusion between classes	Which classes the model tends to mix up (for example, visually similar symptoms on related plants)

Class-level analysis helps explain how the model behaves, not just how often it is correct, and it informs where the model may need improvement or where predictions deserve extra caution.

Per-class numbers are not reproduced here because they are not provided in this documentation. They should be taken from the project's classification report and confusion matrix outputs.

16. Web Application
Overview

The project includes a browser-based application so users can interact with the model without any technical setup.

Technologies
Technology	Role
Python	Primary programming language
Flask	Backend web framework
HTML	Page structure
CSS	Styling and responsive layout
JavaScript	Client-side interaction (upload, preview, drag-and-drop, API calls, dynamic results)
TensorFlow / Keras	CNN model loading and inference
OpenCV	Image analysis (for example, sharpness and color-based checks)
Pillow	Image opening and format handling
NumPy	Numerical array operations
Application features
Image upload
Image preview
Drag and drop
Analyze button
Loading indicator
Prediction result display
Confidence display
Top-3 predictions
Evidence display
Prediction-withheld message
Try-again option
Supported class information
Limitations information
Platform support

The interface is designed to work in:

Desktop browsers
Mobile browsers
17. Application Architecture
┌──────────────────────────────────────────────────────────┐
│                         BROWSER                          │
│   (HTML / CSS / JavaScript: upload, preview, results)    │
└────────────────────────────┬─────────────────────────────┘
                             │  HTTP request (image)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                     FLASK BACKEND                        │
└────────────────────────────┬─────────────────────────────┘
                             ▼
                      File Validation
                             ▼
                       Image Quality
                             ▼
                      Plant Detector
                             ▼
                      Preprocessing
                             ▼
                          CNN
                             ▼
                   Prediction Analysis
                             ▼
                      Decision Layer
                             ▼
                 Accepted  /  Withheld
                             ▼
┌──────────────────────────────────────────────────────────┐
│                         BROWSER                          │
│        (displays result, evidence, or explanation)       │
└──────────────────────────────────────────────────────────┘
Client–server responsibilities
Side	Responsibilities
Browser (client)	Select/drag-and-drop image, show preview, trigger analysis, show loading state, send the image to the backend, render accepted or withheld results, allow retry
Flask backend (server)	Receive the upload, run validation, quality analysis, plant screening, preprocessing, CNN inference, prediction analysis, and the decision logic, then return a structured (JSON) response and handle errors

The heavy computation (model loading and inference, image analysis) happens on the server. The browser handles presentation and interaction.

18. Backend Components

The backend is organized into modular Python files. The responsibilities below describe the intended role of each component without prescribing internal implementation details.

File	Responsibility
app.py	Application entry point. Defines the Flask app and routes, receives uploads, coordinates the pipeline, and returns responses
config.py	Centralized configuration: paths, sizes, class names, thresholds, and other settings
image_validator.py	File-level validation: existence, extension, decodability, dimensions, and size checks
model_loader.py	Loads the trained CNN model so it can be reused for inference
ood_detector.py	Combines heuristic signals (confidence, margin, entropy, consistency, and related signals) for out-of-domain / uncertainty handling
plant_detector.py	Lightweight vegetation/color-based plant vs. non-plant screening
prediction_service.py	Orchestrates preprocessing, CNN inference, and prediction analysis (top-K, confidence, margin, entropy, consistency)
quality_checker.py	Image quality analysis: dimensions, sharpness, brightness, and contrast

These components are part of the current project direction. Exact responsibilities and file boundaries may be refined during Capstone development.

19. Configuration

Configuration is centralized (in config.py) so settings are not scattered through the code. Configuration may include:

Category	Examples
Model	Model path
Input	Image size, number of classes, class names
Upload	Allowed extensions, maximum upload size
Image quality thresholds	Minimum dimensions, blur threshold, brightness limits, contrast limits
Plant detection thresholds	Vegetation ratio threshold(s)
Confidence thresholds	Minimum top-1 confidence
Margin thresholds	Minimum top-1 vs. top-2 gap
Entropy thresholds	Maximum normalized entropy
Consistency thresholds	Required agreement across transformed versions
Top-K settings	Number of top predictions to show (3)
Why centralized configuration helps
Thresholds can be tuned in one place without editing logic throughout the code.
It improves maintainability and readability.
It makes the system easier to adapt for deployment (for example, different paths or limits).
It keeps behavior consistent across modules.

Specific numeric threshold values are intentionally not listed in this README. They are heuristic settings that can be tuned and documented from the actual configuration file.

20. File Validation

File validation is the first gate. It prevents invalid or unsupported inputs from reaching later stages.

Check	Purpose
File existence	Confirms a file was actually provided in the request
Extension validation	Accepts only supported extensions
Image decoding	Confirms the file can actually be opened as an image (not just renamed)
Dimensions	Reads image width and height for later checks
Supported image formats	Limits input to supported formats
Upload size	Enforces a maximum file size

Supported extensions:

.jpg
.jpeg
.png

If validation fails, the system returns a clear error and does not proceed to prediction.

21. Image Quality Analysis

Poor-quality images can cause unreliable CNN predictions, because the visual evidence the model relies on (texture, color, lesion edges) may be missing or distorted.

Measure	Why it matters
Width / Height	Very small images may lack sufficient detail for reliable analysis after resizing
Sharpness	Blurry images lose edge and texture detail that the model depends on
Brightness	Very dark images lose detail; very bright / overexposed images wash out colors and lesions
Contrast	Very low contrast images may lack distinguishable features

If an image falls outside the configured quality ranges, the system may return a Poor Quality result instead of a disease prediction.

These thresholds are heuristic and configurable, not universal standards.

22. Blur Detection
Laplacian variance heuristic

Blur is estimated using the variance of the Laplacian of the image.

The Laplacian operator highlights regions of rapid intensity change (edges).
Sharp images generally contain stronger high-frequency edge information, so the Laplacian response has higher variance.
Blurred images generally have weaker edges, so the variance is lower.
A configurable threshold is used: images with variance below the threshold are treated as potentially too blurry.
Caveat

This is a heuristic, not a universal image-quality standard. A "good" variance value depends on image size, content, texture, and camera. A low-texture but legitimately sharp image may score low, and some blurry images may score higher than expected. The threshold should be viewed as a practical filter, not an absolute judgment.

23. Plant / Non-Plant Screening
Purpose

Before trusting the CNN, the system performs a lightweight screening to check whether the image has plant-like visual characteristics.

How it works (conceptually)

The heuristic uses relationships among RGB channels to estimate how much of the image appears to be plant-related. It considers:

Region type	Rationale
Green vegetation	Healthy leaf tissue is typically green
Yellow regions	Chlorosis or yellowing leaves/tissue
Brown / diseased plant regions	Necrotic or diseased leaf areas often appear brown

Including yellow and brown is important because diseased leaves are not always green, and a green-only check would incorrectly reject many valid diseased-leaf images.

Vegetation ratio

The system computes a vegetation ratio: the proportion of pixels that match the plant-like color rules. A configurable threshold determines whether the image has enough plant-like content to proceed.

Important clarification

This is NOT a complete object detector or segmentation model. It does not identify leaves as objects, and it can be fooled by non-plant objects with similar colors (for example, a green object) or by plants with unusual coloring. It is an additional validation signal, used together with the other signals in the pipeline.

24. CNN Prediction

After validation and preprocessing, the CNN performs inference.

The model receives the validated, preprocessed image.
It outputs probabilities for all 38 classes (softmax).
The top-1 prediction is the highest-probability class.
The top-3 predictions are the three highest-ranked classes.
Illustrative example format

Note: The values below are purely illustrative and are not an actual measured result.

Tomato — Early Blight:   82.4%
Tomato — Late Blight:     8.7%
Tomato — Leaf Mold:       4.1%

The CNN output is the raw material for the later analysis stages. It is not yet the final answer.

25. Confidence Analysis

Top-1 confidence is the probability assigned to the highest-ranked class.

Higher top-1 confidence means the model's probability mass is concentrated on one class.
Lower top-1 confidence means the model is less decisive.
Important caveat

Softmax confidence is not proof of correctness. A model can output high softmax values on images it should not be classifying at all, such as unrelated or out-of-domain images. Softmax confidence also does not equal the real-world probability of being correct.

For this reason, confidence is used as one signal among several, not as the sole basis for acceptance.

26. Prediction Margin

The prediction margin is the gap between the top-1 and top-2 probabilities.

Example comparison
Scenario	Top-1	Top-2	Interpretation
A	Class A = 61%	Class B = 58%	Very small margin. The model is nearly split between two classes. Ambiguous.
B	Class A = 88%	Class B = 7%	Large margin. The model clearly favors one class.

(Illustrative values only. Real distributions need not sum as shown.)

Why margin matters

A top-1 confidence that looks reasonable can still be accompanied by a very close second choice. A small margin indicates that the model could not clearly separate its top candidates, which suggests ambiguity and weakens the case for presenting a single confident answer.

27. Entropy / Uncertainty

Normalized entropy measures how spread out the probability distribution is across the 38 classes, scaled to a standard range.

Distribution shape	Entropy	Interpretation
Focused (most probability on one class)	Lower	Lower uncertainty
Spread (probability distributed across many classes)	Higher	Higher uncertainty
Why it is useful

Entropy captures information that top-1 confidence alone may miss. For example, a distribution that is flat across many classes signals that the model has no strong evidence for any one of them.

Caveat

Entropy is one signal and not a standalone diagnosis mechanism. It should be interpreted together with confidence, margin, consistency, quality, and plant-likelihood.

28. Prediction Consistency

The system can evaluate the same image under small transformations and compare the resulting predictions. Examples of such variants:

Original image
Horizontal flip
Brightness variation
Another brightness variation
Interpretation
Outcome	Meaning
Consistent predictions across variants	Stronger evidence that the prediction is stable
Inconsistent predictions across variants	Possible uncertainty: the prediction may depend on incidental details
Caveat

This is a practical stability check. It does not provide any mathematical robustness guarantee. A prediction can be consistent and still wrong, and passing this check does not prove correctness.

29. OOD / Domain Handling

Out-of-distribution (OOD) or out-of-domain inputs are images that differ from what the model was trained to handle, such as non-plant images, unsupported plants, unsupported diseases, or images with very different visual characteristics from PlantVillage.

This project uses heuristic signals to attempt to identify such cases:

Signal	Contribution
Confidence	Low confidence may indicate the image does not match any known class well
Margin	A small margin may indicate ambiguity between classes
Entropy	High entropy may indicate a spread-out, uncertain distribution
Consistency	Instability under small transformations may indicate unreliable evidence
Image quality	Poor quality may indicate unreliable input
Plant-likelihood	A low vegetation ratio may indicate a non-plant image
Important clarification

This is NOT a perfect or formally guaranteed OOD detector. Some out-of-domain images may still receive confident, consistent predictions and be accepted, and some valid in-domain images may be withheld. The goal is to help reduce misleading predictions, not to eliminate them.

30. Final Decision Engine

The decision engine combines all the signals and produces one of the following outcomes:

#	Outcome	Prediction shown?
1	Prediction Accepted	Yes
2	Poor Quality	No (withheld)
3	Non-Plant	No (withheld)
4	Uncertain	No (withheld)
5	Out of Domain	No (withheld)
Outcome details
1. Prediction Accepted
Meaning: The available signals collectively support the CNN's prediction.
Why: Validation, quality, plant-likelihood, confidence, margin, entropy, and consistency checks were satisfied.
User action: Review the predicted class, confidence, top-3 ranking, and evidence. Treat it as an AI-assisted result, and consult an expert for real decisions.
2. Poor Quality
Meaning: The image does not meet quality requirements (for example, too blurry, too dark, too bright, low contrast, or too small).
Why the system may withhold: Poor-quality images lack reliable visual evidence, so a prediction would not be well-supported.
User action: Retake or upload a sharper, well-lit, clearer image of a single leaf.
3. Non-Plant
Meaning: The image does not appear to contain enough plant-like visual characteristics.
Why the system may withhold: The CNN would still output one of its 38 classes, but that output would be meaningless for a non-plant image.
User action: Upload a clear photograph of a plant leaf.
4. Uncertain
Meaning: The model's output is ambiguous or unstable (low confidence, small margin, high entropy, or inconsistent predictions).
Why the system may withhold: The evidence does not clearly point to one class.
User action: Try a clearer, closer image of a single affected leaf, ideally with a plain background and good lighting.
5. Out of Domain
Meaning: The image appears to fall outside the supported plants/conditions or the model's training domain, according to the heuristic signals.
Why the system may withhold: The image may involve an unsupported species or disease, or differ greatly from the training distribution.
User action: Check the supported classes list. If the plant or disease is not supported, the system cannot reliably analyze it.

The exact mapping between signals and outcomes is controlled by configurable heuristic thresholds and may be refined during Capstone development.

31. Prediction Evidence

When a prediction is accepted, the system can display the supporting signals that led to acceptance:

Evidence	Description
Image dimensions	Width × height of the uploaded image
Sharpness	Blur-related score (Laplacian variance)
Brightness	Brightness measure
Vegetation ratio	Fraction of plant-like pixels
Confidence	Top-1 probability
Margin	Top-1 vs. top-2 gap
Entropy	Normalized entropy measure
Consistency	Agreement across transformed variants
Why this improves transparency

Showing evidence lets users and evaluators see why a result was accepted, instead of treating the system as a black box. It makes the decision process inspectable and gives a more honest picture of how strongly the result is supported.

32. Top-3 Predictions

The interface can show the top three ranked model outputs to provide context about how the model distributed its probability.

Clarification: The top-3 predictions are ranked model outputs, not three confirmed diagnoses. They show which classes the model considered most likely, not three simultaneous conclusions. A close second or third entry can also help users understand when the model was torn between possibilities.

33. User Interface

The interface is designed to make the pipeline's behavior easy to understand.

UI Element	Purpose
Upload	Select an image file
Drag and drop	Drop an image directly onto the upload area
Preview	Show the selected image before analysis
Analyze	Start validation and prediction
Loading indicator	Show that analysis is in progress
Result	Display the final outcome
Confidence	Show the top-1 confidence for accepted predictions
Top-3	Show ranked model outputs
Evidence	Show supporting signals
Withheld state	Show a clear message explaining why no prediction is given
Retry	Let the user try again with another image
Supported classes	Show the list of plants/conditions the system supports
Limitations	Make the system's limits visible

The layout is designed to be responsive, working across desktop and mobile browsers.

34. Project Structure

A representative structure based on the known components:

project/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
├── image_validator.py
├── model_loader.py
├── ood_detector.py
├── plant_detector.py
├── prediction_service.py
├── quality_checker.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── tests/
└── model/
Item	Description
app.py	Flask application entry point
config.py	Central configuration
requirements.txt	Python dependencies
README.md	Project documentation (this file)
.gitignore	Excludes large/unnecessary files (for example, the dataset)
image_validator.py	File validation
model_loader.py	Model loading
ood_detector.py	Heuristic domain / uncertainty signals
plant_detector.py	Plant / non-plant screening
prediction_service.py	Inference and prediction analysis
quality_checker.py	Image quality analysis
templates/index.html	Main page template
static/style.css	Styling
static/script.js	Client-side behavior
tests/	Tests
model/	Trained model location

Note: This structure is representative and the exact layout may evolve during Capstone development. The dataset is not included in the repository.

35. Capstone Scope

This repository represents the Capstone project.

The purpose of the Capstone is not to restart the AI/ML journey from scratch. The earlier AI/ML Foundation work provides the development background: data preparation, machine learning fundamentals, NLP, deep learning, computer vision, CNN training, evaluation, per-class analysis, and early validation/uncertainty ideas. The Capstone productizes and improves that existing work.

Capstone direction

"Develop a reliable AI-assisted plant disease image analysis system that combines CNN-based classification with image validation and uncertainty-aware prediction decisions through a user-friendly web interface."

Capstone activities
Activity	Description	Status
Finalize scope	Fix the project definition, features, limits, and success criteria	In progress
Integrate existing components	Bring the model, validation, quality, plant screening, and uncertainty logic into one coherent system	Planned
Build working prototype	End-to-end functioning application	Planned (next milestone)
Improve UI/UX	Refine the interface, messages, and responsiveness	Planned
Test complete system	Test valid, poor-quality, non-plant, ambiguous, and out-of-domain scenarios	Planned
Deploy	Publish the application (targeting Render)	Planned
Document	Complete the final documentation	In progress
Demonstrate	Final project demonstration	Planned
36. Working Prototype
What the prototype should accomplish

A user should be able to complete the entire flow in a browser:

Open Application
      ↓
Upload Image
      ↓
Preview
      ↓
Analyze
      ↓
Validation
      ↓
CNN
      ↓
Decision Engine
      ↓
Result
Prototype features (PLANNED)
Responsive interface
Upload validation
Image quality checks
Plant / non-plant screening
CNN prediction
Confidence analysis
Margin analysis
Entropy analysis
Consistency analysis
Top-3 predictions
Evidence display
Withheld states with explanations
Error handling

The prototype is the next planned milestone. It is not claimed to be complete at this stage.

37. Deployment
Planned deployment platform: Render

The Capstone deployment phase will target Render. The deployment is planned and has not been completed.

Intended deployment architecture
GitHub Repository
        ↓
      Render
        ↓
 Build Environment
        ↓
   Dependencies
        ↓
   Flask Application
        ↓
   Public Web URL
Deployment considerations
Consideration	Description
Production configuration	Settings appropriate for a deployed environment rather than local development
Dependencies	Pinned and managed through requirements.txt
Build / start configuration	Defining how Render builds and starts the Flask application
Model path	Ensuring the trained model file is available at the path expected by the application
Deployment testing	Verifying the deployed app behaves like the local version
Error checking	Reviewing logs, handling failures, and confirming helpful error messages
Resource limitations	Considering memory, CPU, and startup-time limits, particularly for loading a TensorFlow/Keras model
38. Final Demonstration
Planned demo scenarios
Scenario	Input	Expected behavior
1	Valid plant image	Accepted prediction: class, confidence, top-3, evidence
2	Non-plant image	Prediction withheld (non-plant)
3	Poor-quality image	Prediction withheld (poor quality)
4	Uncertain image	Prediction withheld (uncertain)
Why these scenarios matter

A normal classifier would typically produce a disease label in all four cases. These scenarios show the project's distinguishing value: the system decides whether it has enough evidence before responding, and explains itself when it does not. Scenario 1 shows normal operation, while Scenarios 2–4 show the cautious behavior that sets the project apart.

39. Testing Strategy

The planned testing covers five input categories:

Category	Description	Expected system behavior
Valid plant images	Clear, well-lit leaf images from supported classes	Prediction accepted, with confidence, top-3, and evidence
Poor-quality images	Very blurry, very dark, very bright, low-contrast, or very small images	Prediction withheld with a Poor Quality explanation
Non-plant images	Cars, laptops, people, buildings, random objects	Prediction withheld as Non-Plant (or flagged by other signals)
Ambiguous images	Images where the model's output is split or unstable	Prediction withheld as Uncertain
Unsupported / out-of-domain images	Unsupported plants or diseases, or images very different from training data	Prediction withheld as Out of Domain where heuristic signals detect it

Honest note: No test accuracy figures for the validation/decision layer are claimed here. Because the layer is heuristic, some cases may be misclassified (a bad image accepted or a good image withheld). Observed behavior should be recorded as testing is performed.

40. Limitations
PlantVillage domain: The model is trained on PlantVillage images, which may differ from real-world photos.
38 classes only: Plants and diseases outside the 38 supported classes cannot be correctly identified.
Real-world generalization: Performance on field images with complex backgrounds, lighting, and camera variation may be lower than on the PlantVillage test split.
Plant detector is heuristic: The color-based vegetation check is not an object detector or segmentation model and can be fooled.
Softmax confidence limitation: High confidence does not guarantee correctness and does not reflect real-world probability of being correct.
OOD detection is heuristic: The domain-handling signals are not formally guaranteed and may miss unsupported inputs.
Image-quality thresholds are heuristic: Blur, brightness, and contrast limits are practical settings, not universal standards.
Unsupported diseases: A disease not in the supported list may be mapped to a wrong supported class or withheld.
Unsupported plant species: Other plant species are outside the supported domain.
No professional agricultural diagnosis: Outputs are not expert diagnoses.
No disease severity estimation: The system does not estimate severity (not implemented).
No guarantee of real-world accuracy: The recorded ~92.89% test accuracy applies to the held-out PlantVillage setup used during development.
41. Responsible Use

This is an academic AI/ML Capstone project.

It should not replace:

Agricultural experts
Professional diagnosis
Field inspection
Why withholding is intentional

The system deliberately favors:

"No prediction over an unjustified prediction."

A confident but wrong disease label can mislead a user into inappropriate action or false reassurance. Withholding a prediction when evidence is weak is a deliberate design choice. It trades some convenience (not always giving an answer) for greater caution and transparency. When a prediction is withheld, the user is told why and what to try next, rather than being handed an unsupported label.

42. Technical Skills

The following are skills developed and used through this project and its foundation work. They describe practical project experience, not professional certifications.

Area	Skills
Programming	Python, JavaScript, HTML, CSS
Data Science	Pandas, NumPy, data cleaning, data analysis
Machine Learning	Supervised learning, Linear Regression, classification, model evaluation, train/test split
NLP	Text preprocessing, TF-IDF, Naive Bayes
Deep Learning	TensorFlow, Keras, CNN, softmax, data augmentation, early stopping, model checkpointing
Computer Vision	Image preprocessing, RGB, resizing, blur detection, brightness analysis, contrast analysis, vegetation detection
Model Evaluation	Accuracy, precision, recall, F1, confusion matrix, per-class evaluation, confidence, margin, entropy, consistency
Backend	Flask, file upload, request handling, JSON responses, validation, error handling
Frontend	Responsive HTML, CSS, JavaScript, drag/drop, image preview, API communication, dynamic results
Software Engineering	Modular Python, configuration management, requirements management, testing, Git, GitHub, documentation
Deployment	Web deployment concepts, Render, production configuration, dependency management
43. Future Scope

Everything in this section is FUTURE SCOPE. None of these items are implemented, and none are presented as part of the current system.

Area	Possible improvement
Data	Larger, real-world dataset with field images
Coverage	More plant species and more diseases
OOD detection	Advanced out-of-distribution detection methods
Embedding-based distance	Distance-based OOD scoring in the model's feature space
Deep ensembles	Multiple models to improve uncertainty estimates
Monte Carlo dropout	Dropout-based uncertainty estimation at inference time
Energy-based OOD methods	Energy-score-based detection of unfamiliar inputs
Image segmentation	Isolating leaf regions from backgrounds
Grad-CAM	Highlighting image regions that influenced the prediction
Attention visualization	Visualizing where the model focuses
Saliency maps	Gradient-based importance maps
Disease severity estimation	Estimating how advanced a disease is
Mobile platforms	Android / iOS apps or a Progressive Web App (PWA)
Educational information	Agricultural guidance and educational content about plants and diseases
Continuous model improvement	Ongoing retraining and refinement as new data becomes available
44. Project Roadmap

The project is organized into phases rather than a weekly log.

Phase	Name	What it means	Status
Phase 1	Foundation & Model Development	Building the technical foundation (data preparation, ML fundamentals, deep learning, computer vision) and developing the CNN on PlantVillage with augmentation, early stopping, and checkpointing	Completed (as development background)
Phase 2	Model Evaluation	Evaluating the model on held-out data using accuracy, precision, recall, F1, confusion matrix, classification report, and per-class analysis	Completed (as development background)
Phase 3	Reliability & Validation Layer	Adding file validation, image quality analysis, plant screening, and confidence/margin/entropy/consistency analysis around the CNN	Developed as project basis; integration in Capstone
Phase 4	Web Application	Building the Flask-based browser interface for upload, analysis, and results	Developed as project basis; refinement in Capstone
Phase 5	Capstone Prototype	Integrating all components into one working, tested prototype with polished UI/UX	Planned — next milestone
Phase 6	Deployment	Deploying to a public URL, targeting Render	Planned
Phase 7	Final Polish & Demonstration	Final testing, documentation cleanup, and project demonstration	Planned
45. Final System Architecture
                         USER
                           │
                           ▼
                  Web Application
                           │
                           ▼
                  Flask Backend
                           │
                           ▼
                  File Validation
                           │
                           ▼
                  Image Quality
                           │
                           ▼
              Plant / Non-Plant Check
                           │
                           ▼
                  Image Preprocessing
                           │
                           ▼
                     CNN Model
                           │
                           ▼
                  Prediction Scores
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Confidence           Margin             Entropy
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                  Consistency Check
                           │
                           ▼
                  Domain / OOD Check
                           │
                           ▼
                    Decision Engine
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       ACCEPT PREDICTION         WITHHOLD RESULT
              │                         │
              ▼                         ▼
     Class + Confidence          Reason / Guidance
     + Top-3 + Evidence
46. Final Project Vision
Long-term evolution
Basic CNN Classifier
        ↓
Validated CNN Classifier
        ↓
Uncertainty-Aware Prediction
        ↓
User-Friendly Application
        ↓
Public Deployment
        ↓
Explainable AI
        ↓
Real-World Dataset Expansion
        ↓
Advanced Plant Disease Analysis
Stage	Description	Status
Basic CNN Classifier	CNN trained on PlantVillage	Current basis
Validated CNN Classifier	Validation and quality checks around the CNN	Current basis
Uncertainty-Aware Prediction	Margin, entropy, consistency, and accept/withhold logic	Current basis / Capstone integration
User-Friendly Application	Responsive web interface	Capstone prototype (planned)
Public Deployment	Public web URL	Planned (Render)
Explainable AI	Grad-CAM, saliency, attention visualization	Future scope
Real-World Dataset Expansion	Field images, more species and diseases	Future scope
Advanced Plant Disease Analysis	Severity estimation, richer OOD methods, educational information	Future scope
The central question

The project's main conceptual contribution is the question it asks of every image:

"Does the system have enough evidence to provide a plant disease prediction for this particular image?"

A typical classifier asks, "Which class is most likely?" This project first asks whether a prediction should be given at all.

47. Current Status
Item	Status
Current project phase	Capstone Scope Finalization
Current objective	Finalize the complete Capstone direction and system scope based on the completed AI/ML work
Next planned milestone	Working Prototype
Planned deployment	Render (planned, not yet completed)
Final deliverable	A documented, working, deployable AI-powered plant disease detection and analysis web application with a final demonstration

Deployment has not been completed, and the final prototype has not been declared complete. Items marked planned, in progress, or targeted reflect the current state of the project.

48. Conclusion

This project demonstrates a progression of skills and ideas, from data to a deployable application:

Data Preparation
      ↓
Machine Learning
      ↓
NLP
      ↓
Deep Learning
      ↓
Computer Vision
      ↓
Model Training
      ↓
Evaluation
      ↓
Per-Class Analysis
      ↓
Image Validation
      ↓
Uncertainty Analysis
      ↓
Web Application
      ↓
Capstone Productization
      ↓
Deployment

Early stages built the fundamentals of data handling, machine learning, and text processing. They then moved into deep learning and computer vision, producing a CNN trained and evaluated on PlantVillage. Evaluation and per-class analysis showed where the model is strong and where it is weak. From there, the project recognized that accuracy on a held-out dataset is not the same as reliability on arbitrary user images, and it added image validation and uncertainty analysis. The Capstone phase brings these pieces together as a user-facing application, with deployment planned.

The central goal is not merely to predict a disease. The central goal is:

"To determine whether the system has enough evidence to provide a plant disease prediction for a given image."

A normal classifier always tries to choose a class. This project adds a validation and uncertainty-aware decision layer that can decide not to show a prediction when the evidence is insufficient. That is the project's defining contribution, and it reflects a more careful way to build AI systems that real people will use.

49. Disclaimer

This project is developed as an academic AI/ML Capstone Project.

Predictions are generated by a machine learning model trained on a fixed dataset. Results should not be considered professional agricultural diagnosis or expert advice.

The system may produce incorrect results, especially for:

Unsupported plant species
Unsupported diseases
Poor-quality images
Out-of-domain images
Real-world images significantly different from the training data

For any real agricultural decision, consult qualified agricultural professionals and perform appropriate field inspection.
