# Heart Disease Detection Project

This project implements a complete heart disease detection workflow using:

- A rule-based expert system built using a custom Python Rule Engine
- A machine-learning model (Decision Tree Classifier)
- Data preprocessing and visualization pipelines
- A Streamlit UI for interactive predictions

## Project Deliverables

- Cleaned and preprocessed dataset: `data/cleaned_data.csv`
- Data analysis notebook: `notebooks/data_analysis.ipynb`
- Model training notebook: `notebooks/model_training.ipynb`
- Rule-based expert system with 10+ rules: `rule_based_system/rules.py`
- Tuned decision tree model: `ml_model/decision_tree_model.pkl`
- Accuracy comparison report: `reports/accuracy_comparison.md`

## Folder Structure

text
Heart_Disease_Detection/
│── data/ # Contains the dataset (raw & cleaned)
│ ├── raw_data.csv
│ ├── cleaned_data.csv
│── notebooks/ # Jupyter Notebooks for visualization & preprocessing
│ ├── data_analysis.ipynb
│ ├── model_training.ipynb
│── rule_based_system/ # Rule-based system using a Custom Rule Engine
│ ├── rules.py
│ ├── expert_system.py
│── ml_model/ # Decision Tree implementation
│ ├── train_model.py
│ ├── predict.py
│── utils/ # Helper functions for data cleaning & processing
│ ├── data_processing.py
│── reports/ # Comparison reports and evaluation
│ ├── accuracy_comparison.md
│── ui/ # Streamlit UI for user interaction
│ ├── app.py
│── README.md # Project documentation & setup instructions
│── requirements.txt # List of dependencies

## Setup

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## How To Run

1. Preprocess dataset and save artifacts:

```bash
python utils/data_processing.py
```

2. Generate data-visualization outputs:

```bash
python utils/data_visualization.py
```

3. Train the decision tree model with hyperparameter tuning:

```bash
python ml_model/train_model.py
```

4. Evaluate the expert system:

```bash
python rule_based_system/expert_system.py
```

5. Build model-comparison report:

```bash
python reports/build_accuracy_comparison.py
```

6. Run Streamlit UI:

```bash
streamlit run ui/app.py
```

## Interactive Expert-System Mode

```bash
python rule_based_system/expert_system.py --interactive
```

## Single-Patient ML Prediction

```bash
python ml_model/predict.py
```

## Notes

- The decision tree is evaluated using accuracy, precision, recall, and F1-score.
- The expert system exposes matched rules for explainability.
- `reports/accuracy_comparison.md` summarizes both systems and their tradeoffs.
