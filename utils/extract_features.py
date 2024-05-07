import pandas as pd
import spacy
import json

def extract_features(dataframe, json_file_path = "processed_data/popular_words.json", output_csv_path= "processed_data/extracted_features.csv"):
    """
    Processes a DataFrame to extract features based on NER and saves the results to a CSV file.
    
    Args:
    dataframe (pd.DataFrame): DataFrame containing the text data under a 'text' column.
    json_file_path (str): Path to the JSON file containing lists of popular words.
    output_csv_path (str): Path to save the output CSV file with expanded features.
    """
    # Load the JSON file into a dictionary
    with open(json_file_path, 'r') as file:
        loaded_popular_words = json.load(file)

    # Access lists of popular words
    popular_leagues = loaded_popular_words['Leagues']
    popular_clubs = loaded_popular_words['Clubs']
    popular_homes = loaded_popular_words['Homes']
    popular_players = loaded_popular_words['Players']
    popular_coaches = loaded_popular_words['Coaches']
    popular_nations = loaded_popular_words['Nations']
    popular_continents = loaded_popular_words['Continents']

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Define feature extraction functions
    def capitalize_feature(word):
        return 1 if word[0].isupper() else 0

    def word_shape_feature(word):
        shape = []
        for char in word:
            if char.isupper():
                shape.append('X')
            elif char.islower():
                shape.append('x')
            elif char.isdigit():
                shape.append('d')
            else:
                shape.append(char)
        return ''.join(shape)

    def presence_of_numeric_characters(word):
        return 1 if any(char.isdigit() for char in word) else 0

    def extract_prefix_suffix(word):
        prefixes = [word[:i] for i in range(1, min(5, len(word)))]
        suffixes = [word[-i:] for i in range(1, min(5, len(word)))]
        return prefixes, suffixes

    def in_popular_list(word, category_set):
        return 1 if word in category_set else 0

    # Process text and extract features
    def process_text(text):
        doc = nlp(text)

        features = []
        for token in doc:
            orthographic_features = {
                'capitalized': capitalize_feature(token.text),
                'word_shape': word_shape_feature(token.text),
                'contains_digit': presence_of_numeric_characters(token.text)
            }
            prefixes, suffixes = extract_prefix_suffix(token.text)
            lexical_features = {
                'prefixes': prefixes,
                'suffixes': suffixes,
                'POS': token.pos_
            }
            
            # Syntactic features
            syntactic_features = {
                'dep_tag': token.dep_,  # Dependency tag
                'head_text': token.head.text,  # Text of the token's syntactic head
                'is_chunk': token.ent_iob_  # IOB code for named entities
            }

            # Contextual features
            contextual_features = {
                'prev_word': token.nbor(-1).text if token.i > 0 else None,
                'prev_POS': token.nbor(-1).pos_ if token.i > 0 else None,
                'next_word': token.nbor(1).text if token.i < len(doc) - 1 else None,
                'next_POS': token.nbor(1).pos_ if token.i < len(doc) - 1 else None
            }

            pouplar_features = {
                'in_league': in_popular_list(token.text, popular_leagues),
                'in_club': in_popular_list(token.text, popular_clubs),
                'in_home': in_popular_list(token.text, popular_homes),
                'in_player': in_popular_list(token.text, popular_players),
                'in_coach': in_popular_list(token.text, popular_coaches),
                'in_nation': in_popular_list(token.text, popular_nations),
                'in_continent': in_popular_list(token.text, popular_continents)
            }

            # Combine all features into one dictionary
            combined_features = {**orthographic_features, **lexical_features, **contextual_features, **syntactic_features, **pouplar_features,'token': token.text}
            features.append(combined_features)
        
        return features

    # Apply text processing and explode the DataFrame
    dataframe['features'] = dataframe['Text'].apply(process_text)
    df_exploded = dataframe.explode('features')

    # Convert each dictionary in 'features' column to a Series
    df_exploded = df_exploded['features'].apply(lambda x: pd.Series(x))

    # Save the DataFrame to a CSV file
    df_exploded.to_csv(output_csv_path, index=False)

    print(f"Extracted features are saved to '{output_csv_path}'.")

    return df_exploded
