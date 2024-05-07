import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse

def preprocess_features(df_exploded, output_csv_path='processed_data/processed_features.csv'):
    """
    Processes the exploded DataFrame by flattening list-type features, applying one-hot encoding,
    scaling numeric features, and saves the processed DataFrame to a CSV file.

    Args:
    df_exploded (pd.DataFrame): DataFrame containing extracted features.
    output_csv_path (str): Path to save the output CSV file with processed features.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """
    # Flattening list-type features
    def flatten_features(df, column):
        return df[column].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    # Apply flattening transformation
    df_exploded['prefixes'] = flatten_features(df_exploded, 'prefixes')
    df_exploded['suffixes'] = flatten_features(df_exploded, 'suffixes')

    df_exploded['head_text'] = df_exploded['head_text'].astype(str)
    df_exploded['prev_word'] = df_exploded['prev_word'].astype(str)
    df_exploded['next_word'] = df_exploded['next_word'].astype(str)

    # Define columns for one-hot encoding and for scaling
    categorical_features = ['word_shape', 'POS', 'dep_tag', 'head_text', 'prev_word', 'prev_POS', 'next_word', 'next_POS', 'prefixes', 'suffixes', 'is_chunk']
    numeric_features = ['capitalized', 'contains_digit', 'in_league', 'in_club', 'in_home', 'in_player', 'in_coach', 'in_nation', 'in_continent']

    # Set up the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num','passthrough', numeric_features)
        ])

    # Apply preprocessing to the DataFrame
    features = df_exploded[categorical_features + numeric_features]  # Include all required columns
    processed_features = preprocessor.fit_transform(features)

    # Convert processed features to dense format if they are sparse
    if sparse.issparse(processed_features):
        processed_features = processed_features.toarray()

    # Create a DataFrame from the processed features with names
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_features, columns=feature_names)

    # Save the DataFrame to a CSV file
    processed_df.to_csv(output_csv_path, index=False)
    print(f"Processed features are saved to '{output_csv_path}'.")

    return processed_df
