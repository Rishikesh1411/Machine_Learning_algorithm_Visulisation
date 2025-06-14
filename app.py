"""
IRIS VOTING ENSEMBLE CLASSIFIER APP
Author: RISHIKESH RAJ
Date: 2023
Interactive Machine Learning Tool with Streamlit
"""

# ====================== IMPORTS ======================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ====================== CONFIGURATION ======================
st.set_page_config(
    page_title="Iris Ensemble Classifier by RISHIKESH RAJ",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM STYLING ======================
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover { background-color: #45a049; }
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: white;
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .feature-importance {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .prediction-result {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== AUTHOR INFO ======================
def show_author_info():
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1.5rem; background-color: white; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h3>RISHIKESH RAJ</h3>
        <p>Machine Learning Engineer</p>
        <p style="font-size: 0.9rem; color: #555;">"Building intelligent systems with ensemble methods"</p>
    </div>
    """, unsafe_allow_html=True)

# ====================== MAIN CLASS ======================
class IrisEnsembleClassifier:
    """Interactive Iris Classifier by RISHIKESH RAJ"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.classes = ['setosa', 'versicolor', 'virginica']
        self.features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.is_trained = False
        self.training_history = {}
    
    def load_data(self):
        """Load and preprocess Iris dataset"""
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=self.features)
        y = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return X, y
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare train-test split"""
        X, y = self.load_data()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, estimators, voting='soft'):
        """Train ensemble model with interactive feedback"""
        try:
            base_models = []
            
            # Logistic Regression
            if 'lr' in estimators:
                lr = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    multi_class='multinomial'
                )
                base_models.append(('logistic_regression', lr))
            
            # Random Forest
            if 'rf' in estimators:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=3
                )
                base_models.append(('random_forest', rf))
            
            # SVM
            if 'svm' in estimators:
                svm = SVC(
                    probability=True,
                    random_state=42,
                    kernel='rbf',
                    gamma='scale'
                )
                base_models.append(('svm', svm))
            
            if not base_models:
                raise ValueError("No estimators selected")
            
            # Create voting classifier
            self.model = VotingClassifier(
                estimators=base_models,
                voting=voting,
                n_jobs=-1
            )
            
            # Get data
            X_train, _, y_train, _ = self.prepare_data()
            
            # Train with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f"Training progress: {i}%")
                if i == 100:
                    self.model.fit(X_train, y_train)
            
            self.is_trained = True
            self.training_history = {
                'estimators': [name for name, _ in base_models],
                'voting_type': voting,
                'features': self.features,
                'classes': self.classes
            }
            
            st.balloons()
            st.success("🎉 Model trained successfully!")
            return True
        
        except Exception as e:
            st.error(f"❌ Training error: {str(e)}")
            return False
    
    def evaluate_model(self):
        """Evaluate model with comprehensive metrics"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            _, X_test, _, y_test = self.prepare_data()
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'report': classification_report(y_test, y_pred, target_names=self.classes),
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
            # Display metrics
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("F1 Score", f"{metrics['f1']:.2%}")
            
            # Confusion matrix
            st.subheader("🤔 Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.classes,
                yticklabels=self.classes,
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Classification report
            st.subheader("📝 Classification Report")
            st.code(metrics['report'])
            
            return metrics
        
        except Exception as e:
            st.error(f"❌ Evaluation error: {str(e)}")
            return None
    
    def predict_sample(self, sepal_length, sepal_width, petal_length, petal_width):
        """Make prediction for a single sample"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained yet")
            
            # Create input array
            X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            pred = self.model.predict(X_scaled)[0]
            proba = self.model.predict_proba(X_scaled)[0]
            
            return pred, proba
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            return None, None
    
    def save_model(self, filename):
        """Save trained model to file"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'classes': self.classes,
                'features': self.features,
                'training_history': self.training_history
            }
            
            joblib.dump(model_data, filename)
            return True
        
        except Exception as e:
            st.error(f"❌ Save error: {str(e)}")
            return False

# ====================== STREAMLIT APP ======================
def main():
    """Main application function by RISHIKESH RAJ"""
    
    # Author info
    show_author_info()
    
    # App title
    st.title("🌸 Iris Voting Ensemble Classifier")
    st.markdown("""
    <div style="border-left: 4px solid #4CAF50; padding-left: 1rem; margin-bottom: 2rem;">
        <p>An interactive machine learning application for classifying Iris flowers using ensemble methods.</p>
        <p><strong>Author:</strong> RISHIKESH RAJ | <strong>Dataset:</strong> sklearn Iris</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = IrisEnsembleClassifier()
    
    # Navigation
    st.sidebar.title("🔧 Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["🏠 Home", "🚂 Train Model", "🔮 Make Predictions"],
        index=0
    )
    
    if app_mode == "🏠 Home":
        st.header("Welcome to the Iris Classifier!")
        st.markdown("""
        <div class="model-card">
            <h3>About This App</h3>
            <p>This application uses the famous Iris dataset to demonstrate a voting ensemble classifier.</p>
            
            <h3>Dataset Information</h3>
            <p>The Iris dataset contains measurements for 150 iris flowers from three species:</p>
            <ul>
                <li><strong>Setosa</strong></li>
                <li><strong>Versicolor</strong></li>
                <li><strong>Virginica</strong></li>
            </ul>
            
            <p>Four features are measured for each sample:</p>
            <ol>
                <li>Sepal length (cm)</li>
                <li>Sepal width (cm)</li>
                <li>Petal length (cm)</li>
                <li>Petal width (cm)</li>
            </ol>
            
            <h3>Getting Started</h3>
            <ol>
                <li>Go to <strong>Train Model</strong> to create your ensemble</li>
                <li>Select which algorithms to include</li>
                <li>Evaluate the model performance</li>
                <li>Make predictions on new flower measurements</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data
        st.subheader("Sample Data")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Series(iris.target).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        st.dataframe(df.head())
        
        # Show pairplot
        st.subheader("Data Distribution")
        fig = plt.figure(figsize=(10, 8))
        sns.pairplot(df, hue='species', palette='viridis')
        st.pyplot(fig)
    
    elif app_mode == "🚂 Train Model":
        st.header("🚂 Train Your Ensemble Model")
        
        # Model configuration
        st.subheader("Model Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            use_lr = st.checkbox("Logistic Regression", value=True)
        with col2:
            use_rf = st.checkbox("Random Forest", value=True)
        with col3:
            use_svm = st.checkbox("SVM", value=True)
        
        estimators = []
        if use_lr: estimators.append('lr')
        if use_rf: estimators.append('rf')
        if use_svm: estimators.append('svm')
        
        voting_type = st.radio(
            "Voting Type",
            options=['soft', 'hard'],
            index=0
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=40,
                value=20
            ) / 100
            
            random_state = st.number_input(
                "Random state",
                min_value=0,
                max_value=1000,
                value=42
            )
        
        # Train button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Preparing data..."):
                # Update classifier parameters
                st.session_state.classifier = IrisEnsembleClassifier()
                
                # Train model
                success = st.session_state.classifier.train_model(
                    estimators,
                    voting=voting_type
                )
                
                if success:
                    # Evaluate
                    metrics = st.session_state.classifier.evaluate_model()
                    
                    if metrics:
                        # Save model option
                        st.subheader("💾 Save Model")
                        model_name = st.text_input(
                            "Model name",
                            value="iris_ensemble_model"
                        )
                        
                        if st.button("💾 Save Model"):
                            filename = f"{model_name}.pkl"
                            if st.session_state.classifier.save_model(filename):
                                st.success(f"✅ Model saved as {filename}")
                                
                                # Create download link
                                with open(filename, "rb") as f:
                                    bytes = f.read()
                                
                                st.download_button(
                                    label="⬇️ Download Model",
                                    data=bytes,
                                    file_name=filename,
                                    mime="application/octet-stream"
                                )
    
    elif app_mode == "🔮 Make Predictions":
        st.header("🔮 Make Predictions")
        
        if not st.session_state.classifier.is_trained:
            st.warning("⚠️ No trained model found. Please train a model first.")
        else:
            st.success("✅ Model is ready for predictions!")
            
            st.subheader("Enter Flower Measurements")
            
            col1, col2 = st.columns(2)
            with col1:
                sepal_length = st.slider(
                    "Sepal length (cm)",
                    min_value=4.0,
                    max_value=8.0,
                    value=5.8,
                    step=0.1
                )
                
                sepal_width = st.slider(
                    "Sepal width (cm)",
                    min_value=2.0,
                    max_value=4.5,
                    value=3.0,
                    step=0.1
                )
            
            with col2:
                petal_length = st.slider(
                    "Petal length (cm)",
                    min_value=1.0,
                    max_value=7.0,
                    value=4.0,
                    step=0.1
                )
                
                petal_width = st.slider(
                    "Petal width (cm)",
                    min_value=0.1,
                    max_value=2.5,
                    value=1.2,
                    step=0.1
                )
            
            if st.button("🔮 Predict Species", type="primary"):
                with st.spinner("Analyzing flower..."):
                    pred, proba = st.session_state.classifier.predict_sample(
                        sepal_length, sepal_width, petal_length, petal_width
                    )
                    
                    if pred is not None:
                        # Display prediction
                        st.subheader("Prediction Result")
                        
                        # Prediction card
                        st.markdown(f"""
                        <div class="prediction-result">
                            <h3>Predicted Species: <strong>{pred}</strong></h3>
                            <h4>Confidence Levels:</h4>
                            <ul>
                                <li>Setosa: {proba[0]:.1%}</li>
                                <li>Versicolor: {proba[1]:.1%}</li>
                                <li>Virginica: {proba[2]:.1%}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Visualize probabilities
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=st.session_state.classifier.classes,
                            y=proba,
                            marker_color=['#FF9AA2', '#FFB7B2', '#FFDAC1']
                        ))
                        fig.update_layout(
                            title="Species Probability Distribution",
                            xaxis_title="Species",
                            yaxis_title="Probability",
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ====================== RUN APP ======================
if __name__ == "__main__":
    main()