# Report on the Neural Network Model for Alphabet Soup

## Purpose of the Analysis

The purpose of this analysis is to create a binary classifier using deep learning to predict the success of organizations funded by Alphabet Soup. The goal is to leverage historical data to identify which applicants are most likely to use the funding effectively. This analysis includes data preprocessing, neural network design, model optimization, and evaluation to ensure the highest possible predictive accuracy.

---

## Data Preprocessing

### Target and Features
- **Target Variable**: `IS_SUCCESSFUL` — Indicates whether the funding was used effectively.
- **Features**:
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `STATUS`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`

### Removed Variables
- **Removed Variables**: `EIN` and `NAME` were dropped as they serve only as identifiers and do not provide meaningful predictive power.

### Handling Categorical Variables
- Rare categories in categorical features were grouped under the label "Other" to reduce noise.
- Categorical variables were encoded using `pd.get_dummies()`.

### Splitting and Scaling Data
- The dataset was split into training and testing sets using `train_test_split()`.
- Numerical features were scaled using `StandardScaler` to normalize the data and improve model performance.

---

## Neural Network Design

### Model Architecture (original model)
- **Input Layer**: 36 input features.
- **Hidden Layers**:
  - **First Layer**: 80 neurons with ReLU activation.
  - **Second Layer**: 30 neurons with ReLU activation.
- **Output Layer**: 1 neuron with sigmoid activation for binary classification.

### Model Compilation and Training
- **Loss Function**: `binary_crossentropy`
- **Optimizer**: Adam
- **Metrics**: Accuracy
- **Epochs**: 100

---

## Model Evaluation

### Final Results
- **Loss**: 0.5534
- **Accuracy**: 72.59%

---

## Results

### Data Preprocessing
- **Target Variable**: `IS_SUCCESSFUL`
- **Features**: All columns except `EIN`, `NAME`, and `IS_SUCCESSFUL`.
- **Removed Variables**: `EIN` and `NAME`

### Model Design and Evaluation
- **Neurons, Layers, and Activation Functions**:
  - **Layers**: 2 hidden layers for a balance of complexity and performance.
  - **Neurons**:
    - 80 in the first layer to capture complex patterns.
    - 30 in the second layer.
  - **Activation Functions**: ReLU for hidden layers and sigmoid for the output layer.
- **Performance**: The model achieved an accuracy of 72.59%, below the target of 75%. Despite modifications, further improvements are necessary.

### Steps to Improve Performance
- Increased the number of neurons in each layer.
- Added one more hidden layer.
- Eepochs=120.

---

## Summary of Results

The results of the modified neural network, where the accuracy remained around ~72%, suggest that increasing the complexity of the model did not lead to better performance. This means that the current dataset might not require a more complex model to capture its patterns. The features available in the dataset likely already provide the maximum predictive power achievable with the current setup.

The addition of more nodes and an extra hidden layer didn’t bring significant improvements, which indicates that the dataset might have limitations in terms of the relationships it can capture. Simply increasing the model’s depth or size might not be the right solution here. Instead, the focus should shift to improving the data itself.

It’s possible that the model is starting to overfit, even if this isn’t obvious from the test accuracy. 

Finally, it’s important to revisit the preprocessing steps. Ensuring that all features are scaled, rare categories are grouped into "Other", and potentially irrelevant features are removed could lead to better results. Additionally, creating new features that better represent the relationships in the dataset could unlock higher accuracy.

---

## Recommendation for Alternative Models

To address the current limitations, the following approaches are recommended:

1. **Tree-Based Models**:
   - **Random Forest** or **Gradient Boosting (e.g., XGBoost)**:
     - Handle categorical variables more effectively.
     - Automatically capture non-linear relationships.
     - Random Forest can provide feature importance insights to guide additional feature engineering.

2. **Support Vector Machine (SVM)**:
   - Use an SVM with a non-linear kernel (e.g., RBF) for better performance on datasets with overlapping classes.

3. **Hyperparameter Tuning**:
   - Use Keras Tuner or Grid Search to fine-tune parameters such as the number of neurons, learning rate, batch size, and dropout rate.

---

## Final Thoughts

Although the deep learning model fell short of the target accuracy, it demonstrated the ability to predict success with reasonable accuracy. With further tuning or by leveraging alternative models like XGBoost, performance can likely be improved beyond 75%.
