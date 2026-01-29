# Model Card

## Model Details
This model is a **Random Forest Classifier** trained to predict whether an individual's income exceeds $50K/year using U.S. Census Bureau data. It was implemented in Python using scikit-learn and packaged with a FastAPI endpoint for serving predictions.

Algorithm: Random Forest Classifier

Hyperparameters: 100 trees, max depth 10, random_state=42

## Intended Use
The model is intended for income classification based on demographic and employment features. It can be used for:

- Educational purposes and demonstration of ML pipelines.

- Predicting income brackets in census-like datasets.

It should not be used for real-world high-stakes decisions (e.g., loans, hiring, legal decisions) due to potential bias in training data.

## Training Data
The training data comes from the Census Income dataset, including features such as:
 - `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `hours-per-week`, `native-country`

The target label is `salary`, categorized as `>50K` or `<=50K`.

## Evaluation Data
The model was evaluated on a hold-out test set representing 20% of the dataset. Slice evaluations were also performed for categorical features to check subgroup performance.

## Metrics

The model was evaluated using **Precision, Recall, and F1 Score**. Overall performance on the test dataset:

- **Precision:** 0.7974  
- **Recall:** 0.5385  
- **F1 Score:** 0.6429  

### Performance on Categorical Slices

**Workclass (selected values):**

| Workclass       | Count | Precision | Recall | F1   |
|-----------------|-------|-----------|--------|------|
| Private         | 4,578 | 0.8198    | 0.5135 | 0.6314 |
| Self-emp-inc    | 212   | 0.8431    | 0.7288 | 0.7818 |
| State-gov       | 254   | 0.7576    | 0.6849 | 0.7194 |
| Local-gov       | 387   | 0.7528    | 0.6091 | 0.6734 |
| Federal-gov     | 191   | 0.8409    | 0.5286 | 0.6491 |

**Education (selected values):**

| Education       | Count | Precision | Recall | F1   |
|-----------------|-------|-----------|--------|------|
| Bachelors       | 1,053 | 0.7278    | 0.8200 | 0.7712 |
| Masters         | 369   | 0.8037    | 0.8502 | 0.8263 |
| Doctorate       | 77    | 0.8393    | 0.8246 | 0.8319 |
| Some-college    | 1,485 | 0.9189    | 0.2455 | 0.3875 |

**Sex:**

| Sex    | Count | Precision | Recall | F1   |
|--------|-------|-----------|--------|------|
| Female | 2,126 | 0.7966    | 0.4034 | 0.5356 |
| Male   | 4,387 | 0.7975    | 0.5620 | 0.6594 |

> Full slice metrics for all categorical features are available in `slice_output.txt`.

## Ethical Considerations
- The model may reflect biases present in the training data, including gender, race, and education biases.

- Performance varies across subgroups, which may disadvantage minority groups.

- Users should carefully consider ethical implications before applying predictions to real-world decisions.

## Caveats and Recommendations
- The model is intended for educational purposes and demonstration of ML pipelines.

- Model performance can be improved with hyperparameter tuning, feature engineering, or using more advanced algorithms.

- Always validate model performance on fresh, representative data before deployment.