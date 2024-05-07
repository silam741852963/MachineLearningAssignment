from utils.extract_features import extract_features
from utils.preprocess_features import preprocess_features
from utils.load_data import load_data
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import joblib


def format_feature_names(names):
    formatted_names = []
    for name in names:
        # Remove the 'features__' prefix
        clean_name = name.replace("features__", "")
        
        # Remove the '_1.0' suffix
        clean_name = clean_name.rsplit('_', 1)[0]
        
        formatted_names.append(clean_name)
        
    return formatted_names


def predict_new_data(pipeline_filename='model/svm_pipeline.joblib', label_encoder_filename='model/label_encoder.joblib'):
    """
    Predicts new data using a pre-trained SVM pipeline and a label encoder to convert predicted labels back to original form.
    
    Args:
    raw_data (list): List of raw text data for prediction.
    pipeline_filename (str): Filename of the saved SVM pipeline which includes preprocessing and classifier.
    label_encoder_filename (str): Filename of the saved LabelEncoder.
    
    Returns:
    list: Predicted labels in their original categorical form.
    """

    # Load the full pipeline and label encoder
    pipeline = joblib.load(pipeline_filename)
    label_encoder = joblib.load(label_encoder_filename)

    # Ensure raw_data is a DataFrame with the expected column name
    df = load_data("input/new_data.xlsx")
    df_exploded = extract_features(df, output_csv_path="input/extracted_features_new_data.csv")
    features = preprocess_features(df_exploded, "input/processed_features_new_data.csv")

    
    features_to_add = set(format_feature_names(pipeline.named_steps["preprocessor"].get_feature_names_out()))

    for feature in features_to_add:  # Using set to ensure uniqueness
        if feature not in features.columns:
            features[feature] = 0  # Default value of 0

    # Use the loaded pipeline to make predictions directly on the DataFrame
    # The pipeline should handle all preprocessing, including text vectorization
    predictions = pipeline.predict(features)

    # Convert numeric predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    print("Predictions:", predicted_labels)

    return predicted_labels