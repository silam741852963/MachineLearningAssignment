import pandas as pd
import json

def export_popular_words(league_file='data/league.xlsx', 
                         home_file='data/home.xlsx',
                         club_file='data/club.xlsx',
                         nation_file='data/nation.xlsx', 
                         continent_file='data/continent.xlsx',
                         coach_file='data/coach.xlsx', 
                         player_file='data/player.xlsx', 
                         output_file='processed_data/popular_words.json'):
    """
    Exports popular words from different categories to a JSON file.
    
    Args:
    league_file (str): File path to the Excel file containing leagues, clubs, and homes.
    nation_file (str): File path to the Excel file containing nations and continents.
    coach_file (str): File path to the Excel file containing coach names.
    player_file (str): File path to the Excel file containing player names.
    output_file (str): File path for the output JSON file. Defaults to 'popular_words.json'.
    """
    # Load data from Excel files
    league_df = pd.read_excel(league_file)
    home_df = pd.read_excel(home_file)
    club_df = pd.read_excel(club_file)
    nation_df = pd.read_excel(nation_file)
    continent_df = pd.read_excel(continent_file)
    coach_df = pd.read_excel(coach_file)
    player_df = pd.read_excel(player_file)

    # Extract unique values and convert to lists
    popular_leagues = league_df['League'].drop_duplicates().tolist()
    popular_clubs = club_df['Club'].drop_duplicates().tolist()
    popular_homes = home_df['Home'].drop_duplicates().tolist()
    popular_players = player_df['Player'].drop_duplicates().tolist()
    popular_coaches = coach_df['Coach'].drop_duplicates().tolist()
    popular_nations = nation_df['Nation'].drop_duplicates().tolist()
    popular_continents = continent_df['Continent'].drop_duplicates().tolist()

    # Consolidating into a dictionary
    popular_words = {
        "Leagues": popular_leagues,
        "Clubs": popular_clubs,
        "Homes": popular_homes,
        "Players": popular_players,
        "Coaches": popular_coaches,
        "Nations": popular_nations,
        "Continents": popular_continents
    }

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(popular_words, json_file, indent=4)

    print(f"Popular words successfully exported to {output_file}.")