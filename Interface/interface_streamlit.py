import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Set Streamlit page configuration
st.set_page_config(page_title="Student Grade Prediction", layout="wide")


# Load dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Ensure target column exists
        if 'passed' not in df.columns:
            df['passed'] = df['passed'].apply(lambda x: 1 if x == 'yes' else 0)
        return df
    except FileNotFoundError:
        st.error(f"File {file_path} not found. Please upload or specify the correct path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return pd.DataFrame()


# Load the dataset
file_path = 'datasets/student-data.csv'
data = load_data(file_path)

# Stop the app if dataset is empty
if data.empty:
    st.stop()


# Encode categorical variables
def preprocess_data(df):
    le = LabelEncoder()
    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                           'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    return df


# Preprocess dataset
data = preprocess_data(data)
X = data.drop(columns=['passed'])  # Exclude the target column
y = data['passed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Sidebar Navigation
st.sidebar.title("Student Grade Prediction")
option = st.sidebar.radio(
    "Navigation", ["🏠 Main Menu", "📊 Dataset Analysis", "📈 Model Results", "🎯 Predict Grade"]
)

if option == "🏠 Main Menu":
    st.title("Welcome to the Student Grade Prediction App")
    st.write("### Explore the possibilities of predicting student grades using advanced analytics and models.")
    st.image("https://via.placeholder.com/1000x300.png?text=Student+Grade+Prediction", use_container_width=True)

elif option == "📊 Dataset Analysis":
    st.title("Dataset Analysis")
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Summary Statistics")
    st.table(data.describe())

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 📈 Model Results with Comparison
elif option == "📈 Model Results":
    st.title("Model Results")


    # Load trained models
    @st.cache_data
    def load_model(model_path):
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            st.error(f"Model file {model_path} not found.")
            return None


    # Paths to saved models
    logistic_model_path = "models/logistic_regression_model.joblib"
    knn_model_path = "models/knn_model.joblib"
    svm_model_path = "models/svm_model.joblib"

    # Load models
    logistic_model = load_model(logistic_model_path)
    knn_model = load_model(knn_model_path)
    svm_model = load_model(svm_model_path)

    # Ensure X_test and y_test are available
    if "X_test" not in globals() or "y_test" not in globals():
        st.error("Test data (X_test and y_test) not loaded. Please load them before evaluating models.")
    else:
        # Function to evaluate models
        def evaluate_model(model, X_test, y_test, model_name):
            # Model prediction
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Convert y_test (if categorical) to numerical labels
            if isinstance(y_test[0], str):  # If y_test is categorical (e.g., 'no'/'yes')
                label_map = {'no': 0, 'yes': 1}
                y_test = [label_map[label] for label in y_test]

            # Calculate accuracy and other metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Handle the case when y_pred_proba is None (if the model doesn't support probability prediction)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            cm = confusion_matrix(y_test, y_pred)

            return {
                'model': model_name,
                'accuracy': acc,
                'f1': f1,
                'auc': auc,
                'cm': cm,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

        # Evaluate models
        results = []
        if logistic_model:
            results.append(evaluate_model(logistic_model, X_test, y_test, 'Logistic Regression'))
        if knn_model:
            results.append(evaluate_model(knn_model, X_test, y_test, 'KNN'))
        if svm_model:
            results.append(evaluate_model(svm_model, X_test, y_test, 'SVM'))

        # Display results in a DataFrame
        results_df = pd.DataFrame([
            [res['model'], res['accuracy'], res['f1'], res['auc']]
            for res in results
        ], columns=['Model', 'Accuracy', 'F1 Score', 'ROC AUC'])

        st.write("### Model Comparison Results")
        st.dataframe(results_df)

        # Plot ROC curves
        st.write("### ROC Curve Comparison")
        plt.figure(figsize=(8, 6))

        for res in results:
            if res['y_pred_proba'] is not None:
                # Use pos_label=1 to ensure the correct class is treated as the positive class
                fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'], pos_label=1)
                plt.plot(fpr, tpr, label=f"{res['model']} (AUC={res['auc']:.2f})")

        plt.plot([0, 1], [0, 1], '--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend()
        st.pyplot(plt)

        # Select the best model based on F1 Score
        best_model_row = results_df.sort_values(by='F1 Score', ascending=False).iloc[0]
        st.write("### Best Model Selection")
        st.write(f"Best Model: {best_model_row['Model']}")
        st.write(f"Accuracy: {best_model_row['Accuracy']:.4f}")
        st.write(f"F1 Score: {best_model_row['F1 Score']:.4f}")
        st.write(f"ROC AUC: {best_model_row['ROC AUC']:.4f}")

        # Assign the best model globally
        global best_model
        if best_model_row['Model'] == 'Logistic Regression':
            best_model = logistic_model
        elif best_model_row['Model'] == 'KNN':
            best_model = knn_model
        elif best_model_row['Model'] == 'SVM':
            best_model = svm_model
        else:
            st.error("No valid best model selected.")
            best_model = None

# 🎯 Predict Grade using Best Model
elif option == "🎯 Predict Grade":
    st.title("Predict Grade")

    st.write("### Adjust the Sliders for Prediction 🎛")
    st.markdown(
        """<p style="font-size:16px;">Use the sliders below to set feature values and generate a pass/fail prediction. 
        Hover over the <b>?</b> icons to learn more about each feature.</p>""",
        unsafe_allow_html=True
    )

    # Define tooltips for each feature
    tooltips = {
        'gender': "Gender of the student (0: Male, 1: Female).",
        'race/ethnicity': "Group classification based on ethnicity (e.g., A, B, C, etc.).",
        'parental level of education': "Highest level of education achieved by the student's parent(s).",
        'lunch': "Type of lunch the student receives (0: Free/Reduced, 1: Standard).",
        'test preparation course': "Whether the student completed a test preparation course (0: None, 1: Completed)."
    }

    # Collect user input
    user_data = {}
    col1, col2 = st.columns(2)

    for i, column in enumerate(X.columns):
        with (col1 if i % 2 == 0 else col2):
            min_val, max_val = data[column].min(), data[column].max()
            tooltip = tooltips.get(column, "Feature not documented.")
            user_data[column] = st.slider(
                f"{column} 🎚",
                min_value=float(min_val),
                max_value=float(max_val),
                step=(max_val - min_val) / 100,
                value=(max_val + min_val) / 2,
                help=tooltip  # Tooltip for each slider
            )

    # Prediction button
    if st.button("Predict 🎯"):
        if 'best_model' not in globals() or best_model is None:
            st.error("Best model not defined. Please evaluate models first in the '📈 Model Results' section.")
        else:
            user_df = pd.DataFrame([user_data])
            prediction = best_model.predict(user_df)[0]
            result = "Pass" if prediction == 1 else "Fail"
            st.success(f"Prediction Completed Successfully! The student is predicted to: **{result}**")