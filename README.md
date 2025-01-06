# PWC Challenge
Develop a predictive model (using Python) to forecast an individual's salary based on a given dataset.

## Description
This project focuses on building a predictive model to estimate salaries based on a combination of age, gender, education level, job title, years of experience and description. The pipeline implements data preprocessing, model training, inference, and feature analysis using a modular and scalable approach.

## Installation and Setup Instructions
1. Clone the repository:   
 ```bash
git clone https://github.com/MatiasNaranjo/ChallengePWC.git
```

2. Navigate into the directory:
```bash
cd ChallengePWC
```

3. Create and activate a virtual environment:
```bash
python3 -m venv .venv
```
On macOS/Linux:
```bash   
source .venv/bin/activate   
```
On Windows use:
```bash
.venv\Scripts\activate
```
4. Install dependencies:
   
```bash
pip install -r requirements.txt
```
## Key Features
### Data Preprocessing and Feature Engineering
Merge raw datasets  "people.csv" and "salary.csv") on a common column (id).

Rows with Nan values are dropped.

All features need to be numerical, so gender and education level are transformed into numerical values.

Education level is assigned such that 0 represents the lowest level of education and 2 represents the highest.

Because the job titles were too diverse, the most frequent words were extracted and converted into binary features.

Words in job titles are filtered based on a minimum length and frequency threshold to improve feature representation.

The preprocessed dataset may be large, so is saved to avoid the time-consuming process of re-preprocessing the entire dataset in the future.

Preprocessed dataset is split into training and testing sets, with feature normalization applied to enhance model performance.

### Model Training

#### Neural Network Model (TensorFlow):
Uses a deep learning approach to predict salaries.
Implements early stopping to avoid overfitting.
Trained model is saved for reuse.

#### Baseline Model (Dummy):
A baseline "dummy" model predicts the mean of the target variable (Salary) to benchmark performance.

### Inference and Results Evaluation
Runs the models to generate predictions and compare performance on training and testing datasets. Evaluates and compares key metrics, such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).
Summarizes results for easy interpretation of model accuracy and robustness.

### Feature Importance Analysis
Explains model predictions using SHAP (SHapley Additive exPlanations) to understand the importance of features in influencing salary predictions.