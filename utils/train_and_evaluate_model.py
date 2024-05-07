from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def train_and_evaluate_model(X, y, model_filename='model/svm_pipeline.joblib', label_encoder_filename='model/label_encoder.joblib'):
    """
    Trains an SVM classifier within a pipeline on the provided features and labels, splits the data into training and testing sets,
    evaluates the classifier, prints a classification report, and saves the entire pipeline and label encoder.

    Args:
    X (array-like, sparse matrix): The preprocessed features ready for training.
    y (list or array): The target labels for the classification in their original (textual or categorical) form.
    
    Returns:
    None: Outputs the classification report directly.
    """
    # Convert the target labels
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, label_encoder_filename)  # Save the label encoder

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
    transformers=[
        ('features', OneHotEncoder(handle_unknown='ignore'), X.columns.to_list()),
    ])

    # Create a pipeline with a single step containing the SVM classifier
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(random_state=42, dual=False))  # `dual=False` is typically better for n_samples > n_features
    ])

    # Train the pipeline
    pipeline.fit(X_train, Y_train)

    # Save the trained pipeline
    joblib.dump(pipeline, model_filename)
    print(f"Pipeline saved to {model_filename}")

    # Predict on the test data using the pipeline
    predictions = pipeline.predict(X_test)

    # Evaluate the classifier using zero_division parameter
    print(classification_report(Y_test, predictions, target_names=label_encoder.classes_, zero_division=0))
