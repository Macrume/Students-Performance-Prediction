import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Student Grade Prediction", layout="wide")

# Initialize session state
if 'best_model' not in st.session_state:
    st.session_state['best_model'] = None
if 'label_encoders' not in st.session_state:
    st.session_state['label_encoders'] = {}
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = StandardScaler()


# Load dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'passed' not in df.columns:
            st.error(
                "The dataset does not contain the 'passed' column. Ensure the dataset includes the target variable.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"File {file_path} not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


# Preprocess data
def preprocess_data(df):
    if df is None:
        return None, None

    # Create copies to avoid modifying original data
    df_processed = df.copy()

    # Define categorical and numerical columns
    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                           'nursery', 'higher', 'internet', 'romantic']
    numerical_columns = [col for col in df.columns if col not in categorical_columns + ['passed']]

    # Encode categorical variables
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            st.session_state['label_encoders'][col] = le

    # Scale numerical variables
    if numerical_columns:
        df_processed[numerical_columns] = st.session_state['scaler'].fit_transform(df_processed[numerical_columns])

    return df_processed.drop('passed', axis=1), df_processed['passed']


# Train models function
def train_models(X_train, y_train):
    try:
        models = {

        }

        # Only add pre-trained models if they exist
        model_files = {
            'Logistic Regression': 'models/logistic_regression_model.joblib',
            'KNN': 'models/knn_model.joblib',
            'SVM': 'models/svm_model.joblib'
        }

        for name, path in model_files.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)

        # Train all models
        for name, model in models.items():
            model.fit(X_train, y_train)

        return models
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return None


import streamlit as st


def display_main_menu():
    # Title and header
    st.title("🎓 Welcome to the Student Grade Prediction App")

    # Modern and structured description
    st.markdown("### Explore student performance prediction using machine learning 📊")

    # Add some additional information about the app
    st.markdown("""
    This application helps predict student academic performance using various features:
    - **Student demographic information**: Insights like age, gender, and school.
    - **Family background**: Factors such as family support and parents' education.
    - **Study habits**: Weekly study time and school attendance.
    - **Academic history**: Past academic performance, including class failures.

    """)
    st.write("### Features:")
    with st.expander("Student Demographic Information"):
        st.markdown("""
        This includes features such as age, gender, and the school the student attends.
        - **Age**: The age of the student (typically 15-22).
        - **Gender**: The gender of the student.
        - **School**: The type of school the student attends .
        """)

    with st.expander("Family Background"):
        st.markdown("""
        The support and background provided by the student's family.
        - **Parents' Education Level**: What is the highest level of education completed by the student's parents?
        - **Family Size**: How large is the student's family?
        - **Parental Support**: Does the family provide extra educational support to the student?
        """)

    with st.expander("Study Habits"):
        st.markdown("""
        This section captures the student's study habits and time management.
        - **Study Time**: How many hours per week the student spends on studying.
        - **Absences**: How many days has the student missed school.
        - **Failures**: Has the student experienced any past class failures?
        """)

    with st.expander("Academic History"):
        st.markdown("""
        The academic performance of the student in previous periods.
        - **Previous Grades**: The student's grades from previous periods (e.g., G1, G2).
        - **Final Grade (G3)**: The target output which is the final grade prediction.
        """)


def display_dataset_analysis(data):
    st.title("Dataset Analysis")

    # Display basic statistics
    st.subheader("Dataset Overview")
    st.dataframe(data.head())
    st.write(f"**Dataset Shape:** {data.shape}")

    # Missing values analysis
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0] if missing_values.any() else "No missing values found")

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Create visualizations with error handling
    try:
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(data.corr(), cmap='coolwarm', annot=False)
        st.pyplot(fig)
        plt.close()

        # Target Distribution
        st.subheader("Distribution of Target Variable: Passed")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='passed')
        st.pyplot(fig)
        plt.close()

        # Feature Relationships
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.subheader("Feature Relationships")
            selected_feature = st.selectbox("Select feature to analyze:", numerical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(data=data, x='passed', y=selected_feature)
            st.pyplot(fig)
            plt.close()

    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")


def display_model_results(X_train, X_test, y_train, y_test):
    st.title("Model Performance Analysis")

    # Calculate and store feature correlations
    # We'll use training data to avoid data leakage
    X_train_df = pd.DataFrame(X_train, columns=X_test.columns)
    y_train_series = pd.Series(y_train)
    correlations = X_train_df.apply(lambda x: x.corr(y_train_series))
    st.session_state['feature_correlations'] = correlations

    # Train models
    models = train_models(X_train, y_train)
    if models is None:
        return

    # Create tabs for different types of results
    tabs = st.tabs(["Model Comparison", "Detailed Metrics", "ROC Curves"])

    with tabs[0]:
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, y_pred_proba)
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

    with tabs[1]:
        for name, model in models.items():
            st.subheader(f"{name} Metrics")
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.write("Confusion Matrix")
                st.write(pd.DataFrame(
                    cm,
                    columns=['Predicted Negative', 'Predicted Positive'],
                    index=['Actual Negative', 'Actual Positive']
                ))

            with col2:
                st.write("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write(pd.DataFrame(report).transpose())

    with tabs[2]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        st.pyplot(fig)
        plt.close()

    # Save best model
    best_model_name = results_df.loc[results_df['F1 Score'].idxmax(), 'Model']
    st.session_state['best_model'] = models[best_model_name]
    st.success(f"Best performing model: {best_model_name}")


def display_prediction_interface(X):
    st.title("Grade Prediction")

    # Check if the model and feature correlations are available
    if st.session_state['best_model'] is None:
        st.warning("Please run the model evaluation first to train the model!")
        return

    # Ensure feature correlations are available
    if 'feature_correlations' not in st.session_state:
        st.error("Please run the model evaluation to calculate feature correlations first!")
        return

    # Get the top 10 most correlated features
    correlation_series = abs(st.session_state['feature_correlations'])
    top_features = correlation_series.nlargest(10).index.tolist()

    st.write("### Enter Student Information")
    st.info(
        "You only need to fill in values for the 10 most influential features. Other features will use their average values automatically.")

    # Attribute descriptions
    attribute_descriptions = {
        "school": "Student's school (binary: 'GP' - Gabriel Pereira, 'MS' - Mousinho da Silveira)",
        "sex": "Student's sex (binary: 'F' - female, 'M' - male)",
        "age": "Student's age (numeric: 15 to 22)",
        "address": "Student's home address type (binary: 'U' - urban, 'R' - rural)",
        "famsize": "Family size (binary: 'LE3' - <= 3, 'GT3' - > 3)",
        "Pstatus": "Parent's cohabitation status (binary: 'T' - living together, 'A' - apart)",
        "Medu": "Mother's education level (numeric: 0 - none, 1 - primary, 2 - 5th-9th grade, 3 - secondary, 4 - higher)",
        "Fedu": "Father's education level (numeric: 0 - none, 1 - primary, 2 - 5th-9th grade, 3 - secondary, 4 - higher)",
        "Mjob": "Mother's job (nominal: 'teacher', 'health', 'services', 'at_home', 'other')",
        "Fjob": "Father's job (nominal: 'teacher', 'health', 'services', 'at_home', 'other')",
        "reason": "Reason for choosing this school (nominal: 'close to home', 'reputation', 'course preference', 'other')",
        "guardian": "Student's guardian (nominal: 'mother', 'father', 'other')",
        "traveltime": "Home to school travel time (numeric: 1 - <15 min, 2 - 15-30 min, 3 - 30 min to 1 hour, 4 - >1 hour)",
        "studytime": "Weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, 4 - >10 hours)",
        "failures": "Number of past class failures (numeric: 1-2, else 4)",
        "schoolsup": "Extra educational support (binary: 'yes' or 'no')",
        "famsup": "Family educational support (binary: 'yes' or 'no')",
        "paid": "Extra paid classes (binary: 'yes' or 'no')",
        "activities": "Extra-curricular activities (binary: 'yes' or 'no')",
        "nursery": "Attended nursery school (binary: 'yes' or 'no')",
        "higher": "Wants to pursue higher education (binary: 'yes' or 'no')",
        "internet": "Internet access at home (binary: 'yes' or 'no')",
        "romantic": "In a romantic relationship (binary: 'yes' or 'no')",
        "famrel": "Quality of family relationships (numeric: 1 - very bad to 5 - very good)",
        "freetime": "Free time after school (numeric: 1 - very low to 5 - very high)",
        "goout": "Going out with friends (numeric: 1 - very low to 5 - very high)",
        "Dalc": "Workday alcohol consumption (numeric: 1 - very low to 5 - very high)",
        "Walc": "Weekend alcohol consumption (numeric: 1 - very low to 5 - very high)",
        "health": "Current health status (numeric: 1 - very bad to 5 - very good)",
        "absences": "Number of school absences (numeric: 0 to 93)",
        "G1": "First period grade (numeric: 0 to 20)",
        "G2": "Second period grade (numeric: 0 to 20)",
        "G3": "Final grade (numeric: 0 to 20, output target)"
    }

    # Display correlation information
    st.write("### Feature Importance")
    correlation_df = pd.DataFrame({
        'Feature': top_features,
        'Correlation': correlation_series[top_features].round(3)
    })
    st.dataframe(correlation_df)

    # Create a form for input
    with st.form("prediction_form"):
        # Split features into two columns for better layout
        col1, col2 = st.columns(2)
        user_input = {}

        # Initialize all features with mean values
        for feature in X.columns:
            user_input[feature] = X[feature].mean()

        # Create inputs only for top features
        for i, feature in enumerate(top_features):
            with col1 if i < 5 else col2:
                st.write(f"**{feature}:** {attribute_descriptions.get(feature, 'No description available.')}")

                # Display feature correlation value
                correlation_value = correlation_series[feature]
                st.write(f"**Correlation:** {correlation_value:.3f}")

                if feature in st.session_state['label_encoders']:
                    # Categorical feature with label encoding
                    options = st.session_state['label_encoders'][feature].classes_
                    value = st.selectbox(
                        f"{feature} (categorical)",
                        options,
                        help=f"Correlation with outcome: {correlation_value:.3f}"
                    )
                    user_input[feature] = st.session_state['label_encoders'][feature].transform([str(value)])[0]
                else:
                    # Numerical feature, use slider for better interaction
                    min_value, max_value = X[feature].min(), X[feature].max()
                    value = st.slider(
                        f"{feature} (numerical)",
                        min_value=min_value,
                        max_value=max_value,
                        value=float(X[feature].mean()),
                        step=0.1,
                        help=f"Average: {X[feature].mean():.2f}, Correlation: {correlation_value:.3f}"
                    )
                    user_input[feature] = value

        # Submit button for prediction
        submitted = st.form_submit_button("Predict")

        if submitted:
            try:
                # Prepare the input data for prediction
                input_df = pd.DataFrame([user_input])
                prediction = st.session_state['best_model'].predict(input_df)[0]
                probability = st.session_state['best_model'].predict_proba(input_df)[0][1]

                # Display the results in a user-friendly manner
                st.markdown("---")
                st.markdown("### Prediction Results")

                col1, col2, col3 = st.columns([1, 1, 2])

                with col1:
                    st.metric(
                        "Predicted Outcome",
                        "Pass" if prediction == 1 else "Fail"
                    )
                with col2:
                    st.metric(
                        "Probability of Passing",
                        f"{probability:.1%}"
                    )
                with col3:
                    st.info("Prediction based on the 10 most influential features. Other features use average values.")

                # Display the feature values used for prediction
                st.markdown("### Feature Values Used for Prediction")

                prediction_details = pd.DataFrame({
                    'Feature': top_features,
                    'Value': [user_input[f] for f in top_features],
                    'Correlation': correlation_series[top_features]
                })

                # Highlight correlations for better readability
                st.dataframe(prediction_details.style.background_gradient(subset=['Correlation'], cmap='RdYlBu'))

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")


def main():
    # Add error handling for data loading and processing
    try:
        # Load the data
        data = load_data('data/processed_data.csv')
        if data is None:
            st.stop()

        # Preprocess the data
        X, y = preprocess_data(data)
        if X is None or y is None:
            st.stop()

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Sidebar navigation with modern styling
        st.sidebar.title("Navigation 🧭")

        # Sidebar elements to navigate the app
        st.sidebar.markdown("### Explore the app:")
        pages = {
            "🏠 Main Menu": display_main_menu,
            "📊 Dataset Analysis": lambda: display_dataset_analysis(data),
            "📈 Model Results": lambda: display_model_results(X_train, X_test, y_train, y_test),
            "🎯 Predict Grade": lambda: display_prediction_interface(X)
        }

        # Add custom icon/emoji styling for the sidebar options
        selection = st.sidebar.radio(
            "Go to",
            list(pages.keys()),
            index=0,  # Default selection is Main Menu
            key="sidebar_navigation"
        )

        # Display the selected page
        pages[selection]()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and try again.")


# Main entry point
if __name__ == "__main__":
    main()
