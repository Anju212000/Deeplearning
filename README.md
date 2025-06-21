Chronic Obstructive Pulmonary Disease (COPD) remains one of the leading causes of morbidity worldwide, often going undiagnosed until advanced stages. This project presents a deep learning-based solution to detect COPD using electrocardiogram (ECG) signals, offering a low-cost and non-invasive diagnostic alternative.

The study leverages the PTB-XL ECG dataset, a large publicly available collection of annotated 12-lead ECG recordings. A Bidirectional Long Short-Term Memory (BiLSTM) neural network was developed to analyze temporal features within the ECG signals, capturing bidirectional dependencies that are often overlooked by traditional models.

🔍 Key Highlights:

✅ Data preprocessing: ECG signal resampling, normalization, segmentation

✅ Label extraction and binary classification setup (COPD vs. non-COPD)

✅ Deep learning pipeline built using TensorFlow and Keras

✅ BiLSTM model architecture designed to learn temporal patterns in ECG sequences

✅ Model evaluation with accuracy, loss, and confusion matrix analysis

✅ Results visualized with training/validation curves and diagnostic insights

💻 Tech Stack: Python, Pandas, NumPy, TensorFlow/Keras, Matplotlib
📚 Dataset: PTB-XL (PhysioNet 12-lead ECG Database)
📊 Approach: Time-series modeling with BiLSTM

This work contributes to the growing intersection of cardiovascular signal analysis and pulmonary disease diagnostics, highlighting the potential of AI in preventive healthcare.
