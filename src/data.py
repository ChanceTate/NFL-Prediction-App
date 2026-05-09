import pandas as pd
import nflreadpy as nfl

def load_data():
    #Read the CSV file into pandas with utf-8 encoding, we skip the first headers
    #due to it leading to messy data
    df = pd.read_csv("lamar-jackson-stats-2025.csv", encoding="utf-8", header=1)
    df= df[df["Date"].str.contains("/")]

    #Renaming unnamed column to a more readable name
    df.rename(columns={"Unnamed: 6": "home_away"}, inplace=True)

    #Lambda function to replace home_away column with a home_game column
    #0 if game is away and 1 if game is home
    df["home_game"] = df["home_away"].apply(lambda x: 0 if x == "@" else 1)

    #Drop the original column with bad data
    df = df.drop(columns=["home_away"])

    #Moving the new home_game column to be between team and opp for better
    #Readability
    df.insert(0, "Name", "Lamar Jackson")
    col = df.pop("home_game")
    df.insert(6, "home_game", col)

    print(df)
    return df
load_data()



