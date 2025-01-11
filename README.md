# Skin-Disease-Detection-Using-CNN

![UI](https://github.com/user-attachments/assets/3f43b2f4-23ac-41e8-84db-40e33a1daf33)

## Overview

Dermatology is one of the most challenging domains in medical diagnosis due to the complexity and variability of skin conditions. Traditional diagnosis often relies on a dermatologist's experience and extensive testing, leading to varying results and time delays. To address these challenges, this project proposes an **Automated Image-Based Dermatology Diagnosis System** that leverages machine learning techniques to analyze and classify skin diseases efficiently and accurately.

This system utilizes advanced image processing and classification methods to automate the recognition of skin diseases, reducing dependency on manual processes and expertise. By integrating **Convolutional Neural Networks (CNNs)** and optimized image preprocessing techniques, this application ensures reliable, fast, and consistent results.

---

## Key Features

1. **Automated Skin Disease Detection**  
   The system analyzes skin lesion images, extracts features, and classifies them into predefined disease categories using deep learning.

2. **Image Preprocessing & Enhancement**  
   - Removal of noise using advanced filters.  
   - Conversion to grayscale for uniformity (RGB2Gray scaling).  
   - Feature extraction using edge detection techniques like **Canny Edge Detection**.  

3. **Deep Learning for Classification**  
   - **Convolutional Neural Networks (CNNs)** for feature extraction and classification.  
   - **Softmax Classifier** for final diagnosis predictions.  
   - High accuracy through robust training on a diverse dataset.

4. **User-Friendly Interface**  
   - Designed with **Tkinter GUI** for an intuitive and accessible user experience.  
   - Allows users to upload skin lesion images and receive diagnosis results instantly.

5. **Scalability and Speed**  
   - Provides faster results compared to traditional diagnostic methods.  
   - Scalable to integrate additional skin conditions and datasets in the future.

---

## Benefits

- **Improved Diagnostic Accuracy**: Machine learning algorithms outperform traditional visual inspections by learning from large datasets.  
- **Faster Results**: Automated processing significantly reduces diagnosis time.  
- **Accessibility**: Makes expert-level dermatological diagnosis accessible even in remote areas where specialists may not be available.  
- **Cost-Effective**: Reduces dependency on expensive tests and prolonged consultations.  

---

## Tools & Technologies

- **Programming Language**: Python  
- **Machine Learning Frameworks**: TensorFlow, Keras, Scikit-learn  
- **GUI Development**: Tkinter  
- **Image Processing Libraries**: OpenCV, Numpy  
- **Visualization**: Matplotlib  

---

## Workflow

1. **Image Input**:  
   User uploads an image of the skin lesion via the GUI interface.

2. **Image Preprocessing**:  
   - Noise removal using OpenCV filters.  
   - Conversion of the image to grayscale.  
   - Enhancement using edge detection (e.g., Canny Edge Detection).  

3. **Feature Extraction**:  
   CNN processes the preprocessed image, extracting unique features that define the skin condition.

4. **Classification**:  
   - The system applies the Softmax classifier for final disease prediction.  
   - Outputs a diagnosis report with probability scores for each disease category.

5. **Result Display**:  
   The GUI displays the predicted disease, confidence score, and potential next steps for the user.

---

## Future Enhancements

- **Real-Time Diagnosis**: Integration with mobile apps for real-time skin condition assessments.  
- **Expanded Dataset**: Incorporation of more diverse skin types and conditions to improve system robustness.  
- **Explainable AI**: Adding interpretability features to explain the decision-making process of the model to users.  
- **Integration with Healthcare Systems**: Seamlessly connect with electronic health records (EHRs) for comprehensive patient management.  
- **Multi-Modal Analysis**: Combine text-based patient histories with image data for holistic diagnosis.  

---

## Dataset

The system relies on publicly available datasets such as:  
- **ISIC (International Skin Imaging Collaboration)**  
- **HAM10000 Dataset**  
These datasets provide diverse and well-annotated skin lesion images, ensuring accurate training and evaluation.

---

## Contributions

Contributions to enhance this project are welcome! Feel free to submit issues, feature requests, or pull requests via GitHub.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Clone the repo

Clone the repository:
   ```bash
   git clone https://github.com/UwaisMansuri22/Skin-Disease-Detection-Using-CNN.git
   cd your-repo-folder
