# 🌥️ Cloud Coverage Detection (CCD)

Predicting cloud coverage is essential for weather forecasting, solar energy optimization, and beyond. This project focuses on analyzing cloud coverage data and predicting it for 15m, 25m, and 30m intervals using advanced time-series modeling techniques. Here's how we tackled this fascinating challenge! 🚀

---

## 📂 Dataset Details  
- **Provided by**: IIT Roorkee  
- **Type**: Cloud images and corresponding CSV files with minute-separated data entries.  
- **Duration**: Data from **1 Jan to 30 Nov**, recorded at 10-minute intervals (morning to evening).  
- **Preprocessed Data**:  
  - After interpolation and null removal:  
    - **Days covered**: 335  
    - **Interval**: 1-minute-separated entries.  

---

## 🛠️ Initial Approach  

### 🔍 Idea:  
- Use 10-minute interval data.  
- Combine features into a time-series model.  
- Train a **CNN** for cloud coverage prediction.  

---

## 🔧 Dataset Preprocessing  
We explored different preprocessing methods:  

1. **Raw Data**:  
   - Interpolated missing values.  
   - Converted dates and removed nulls.  

2. **Date-Time Systems**:  
   - Added running values for time of year and day as features.  

3. **Data Sampling with Sliding Window**:  
   - Used **PyTorch DataLoader** to sample:  
     - **Gap size**: 10 minutes.  
     - **Sequence length**: 30 (30 minutes).  
     - **Input features**: 17 (columns of the dataframe).  

4. **Target Feature Engineering**:  
   - **Total Cloud Coverage** shifted by 15m, 25m, 35m to create target columns.  

5. **Handling Missing Values**:  
   - Interpolated gaps.  
   - Dropped rows with NaNs at the dataset's end.  

6. **Normalization**:  
   - Applied **Standard Scaler** and **Min-Max Scaler**.  
   - Scaled target values to a 0–1 range for smooth model training.  

7. **Train-Test Split**:  
   - Last 20% rows were used as validation data.  

---

## 🎥 Data Visualization  
We created **short videos** to visualize daily cloud coverage percentages. Each day’s video is:  
- **Resolution**: 400x400x3  
- **FPS**: 5  

---

## 🚀 Model Development  

### 🌀 Initial RNN Approach  
- Input: Raw data with minimal preprocessing.  
- Results: **Decent**, as expected from a baseline RNN.

---

### 💧 Liquid Neural Networks (LNN)  
**Major Focus!**  

#### 💡 Initial Attempt:  
- Used **NCPS (Neuron Connection Probability Sampling)** in PyTorch.  
- Input: 6 hours of data, passed through:  
  - Linear layers for dimension control.  
  - Inter-Neuron Reservoir.  
- **Issue**: Extremely volatile results.  

#### 🔄 Optimized Approach:  
- Reduced reservoir neurons from 30 ➡️ 24.  
- Used **ReLU activation** and **fewer linear layers**.  
- Changed input: Passed **30-minute data** instead of 6 hours.  
- Results: Noticeable improvement!  

---

### 🔄 LSTM Model  
- **Preprocessing**:  
  - Forward and backward fill for missing values.  
  - Min-Max scaling for feature normalization.  

- **Architecture**:  
  - **Input Layer**  
  - 2 **LSTM layers** (50 units each).  
  - **Dropout layers** for overfitting prevention.  
  - **Dense layer** with Sigmoid activation.  

- **Compilation**:  
  - Optimizer: **Adam**  
  - Loss: **Mean Squared Error (MSE)**  

---

## 📊 Results  

| Model           | R² (15m) | R² (25m) | R² (30m) | Aggregate Metric |  
|------------------|----------|----------|----------|------------------|  
| **RNN (1 Layer)**  | 0.6973   | 0.7155   | 0.7196   | 0.7070           |  
| **RNN (3 Layers)** | 0.8222   | 0.8222   | 0.9910   | 0.8701           |  
| **LNN (Optimized)**| 0.9747   | 0.9719   | 0.9594   | 0.9699           |  
| **LSTM**          | -        | -        | -        | 0.8775           |  

📈 **Wandb Logs**:  
- [Initial LNN Results](https://api.wandb.ai/links/roadkill/4nytxm5x)  
- [Final LNN Results](https://wandb.ai/roadkill/CCD/reports/CCD-Cloud-Coverage-Dataset-CCD-prediction--Vmlldzo4NDE5ODQ3)  

---

## 🌟 Future Scope  

### NCCLNet (Naive Continuous Convolution Liquid Network)  
- Combines image and dataframe data for predictions.  
- Uses **Synthetic Minority Oversampling Technique (SMOTE)** to generate additional feature maps for odd-minute data.  

### Transformer Models  
- Use transformers as **feature extractors** for image data.  

### ARIMA  
- Experiment with **AutoRegressive Integrated Moving Average** models for time-series prediction.  

### Wind as a Vector  
- Represent wind as a vector (speed + direction).  
- Predict new cloud movements based on wind patterns.  

---

## 🤝 Contributing  
Want to contribute? Feel free to fork the repo and make a pull request!  

---
