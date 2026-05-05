from data import load_data

def build_features(df):

    df["home_game"] = df["home_game"].astype(int)

    X = df[[
        "home_game",
        "Cmp%",
        "Yds",
        "TD",
        "Int"
        "Y/A",
        "Rate"
    ]]

    return X