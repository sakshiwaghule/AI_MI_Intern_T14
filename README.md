# AI_MI_Intern_T14
# Model Benchmarking and Selection Pipeline
This project implements a comprehensive machine learning pipeline to classify model architectures based on experimental performance metrics. The pipeline evaluates several classical algorithms to determine which can best predict model types using historical metadata.
# 🚀 Features
* **Multi-Model Training:** Implementation of Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).
* **Automated Evaluation:** Generates Accuracy, Precision, Recall, and F1-Score for every model.
* **Generalization Analysis:** Compares Training vs. Testing accuracy to detect overfitting.
* **Visualization:** Produces a visual performance comparison chart.
* **Artifact Management:** Saves the best-performing model along with its preprocessing scalers for future inference.
# 📊 Methodology
# 1. Data Preprocessing
* **Feature Selection:** Utilizes `learning_rate`, `batch_size`, `epochs`, and performance metrics (`accuracy`, `f1_score`, etc.).
* **Scaling:** Applies `StandardScaler` to normalize numerical features, ensuring compatibility with distance-based models like SVM.
 **Label Encoding:** Transforms categorical model names into numerical formats.
# 2. Performance Comparison
The models were evaluated using weighted metrics to account for class distribution. Below is the final comparison:
| Model | Accuracy | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| **Random Forest** | 0.1500 | 0.2167 | 0.1500 | 0.1768 |
| **Decision Tree** | 0.1000 | 0.1625 | 0.1000 | 0.1222 |
| **SVM** | 0.0500 | 0.0143 | 0.0500 | 0.0222 |
| **Logistic Regression** | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
# 3. Best Model Selection
The **Random Forest** model was selected as the champion model based on the business requirement of maximizing the **F1-Score**. While the current feature set provides low predictive power, Random Forest showed the highest capacity to capture underlying patterns.
# 📁 Output Files
* `model_predictions.csv`: Contains actual vs. predicted values for the test set.
* `model_comparison_table.csv`: A tabular summary of all evaluation metrics.
* `model_performance_comparison.png`: A bar chart visualizing model scores.
* `best_model_artifacts.pkl`: The serialized best model, scaler, and label encoder.
# 🛠️ Requirements
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `joblib`
