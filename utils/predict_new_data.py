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


import pandas as pd
import joblib

def predict_new_data(pipeline_filename='model/svm_pipeline.joblib', 
                     label_encoder_filename='model/label_encoder.joblib', 
                     input_data_filename='input/new_data.xlsx',
                     extracted_features_path='output/extracted_features_new_data.csv',
                     processed_features_path='output/processed_features_new_data.csv',
                     predictions_output_path='output/predictions_new_data.csv'):
    """
    Predicts new data using a pre-trained SVM pipeline and a label encoder to convert predicted labels back to original form,
    while saving intermediate steps and final predictions.

    Args:
    pipeline_filename (str): Path to the saved SVM pipeline which includes preprocessing and classifier.
    label_encoder_filename (str): Path to the saved LabelEncoder.
    input_data_filename (str): Path to the input data file.
    extracted_features_path (str): Path to save extracted features CSV.
    processed_features_path (str): Path to save processed features CSV.
    predictions_output_path (str): Path to save final predictions CSV.
    """

    # Load the full pipeline and label encoder
    pipeline = joblib.load(pipeline_filename)
    label_encoder = joblib.load(label_encoder_filename)

    # Load and process data
    df = pd.read_excel(input_data_filename)
    df_exploded = extract_features(df, output_csv_path=extracted_features_path)
    features = preprocess_features(df_exploded, processed_features_path)

    # Ensure all required features are present
    required_features = set(format_feature_names(pipeline.named_steps['preprocessor'].get_feature_names_out()))
    for feature in required_features:
        if feature not in features.columns:
            features[feature] = 0  # Add missing features with default value of 0

    # Make predictions and convert to original labels
    predictions = pipeline.predict(features)
    predicted_labels = label_encoder.inverse_transform(predictions)
    
    df_exploded['predicted_labels'] = predicted_labels  # Append predictions to the input data for full traceability

    # Save the DataFrame with predictions
    df_exploded.to_csv(predictions_output_path, index=False)

    print("Predictions saved to:", predictions_output_path)
    return df_exploded  # Optionally return the DataFrame for further usage