from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

def train_and_evaluate_model(features, labels):
    """
    Trains an SVM classifier on the provided features and labels, splits the data into training and testing sets,
    and evaluates the classifier, printing a classification report.

    Args:
    features (array-like, sparse matrix): The preprocessed features ready for training.
    labels (list or array): The target labels for the classification.
    
    Returns:
    None: Outputs the classification report directly.
    """
    # Convert the target labels
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(features, Y_encoded, test_size=0.2, random_state=42)

    # Initialize and train the SVM classifier
    svm_classifier = LinearSVC(random_state=42)
    svm_classifier.fit(X_train, Y_train)

    # Predict on the test data
    predictions = svm_classifier.predict(X_test)

    # Evaluate the classifier
    print(classification_report(Y_test, predictions, target_names=label_encoder.classes_))
