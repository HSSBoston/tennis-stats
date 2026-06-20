import pandas as pd
from eventparser import classifyEvent, EVENT_TYPES

# Adds an event label and calculates the change in game-win probability, delta V, 
# caused by each point.
#   df: Point-by-point dataset that has been created by winprob.computeV().
#   vDict: Maps each game state to the server's game-win probability.
#   
def computeDeltaGameWinExpectancy(df: pd.DataFrame, vDict: dict) -> pd.DataFrame:
    df = df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
        # Sort rows by "match_id" and then "Pt"
        # drop=True: Discard old index numbers
    
    # Group all point rows according to their match IDs and then game numbers.
    # Select the "Pts" column and move the Pts values upward by one row. Therefore,
    # each row receives the next row's value. For example:
    # Before:
    #   Pt   Pts
    #   1    0-0
    #   2    15-0
    #   3    30-0
    #   4    40-0
    # After:
    #   Pt   Pts   next_state
    #   1    0-0   15-0
    #   2    15-0  30-0
    #   3    30-0  40-0
    #   4    40-0  NaN
    df["next_state"] = df.groupby(["match_id", "Gm#"])["Pts"].shift(-1)

    # After:
    #   Pt   Pts   next_state   V_before  V_after
    #   1    0-0   15-0         0.66      0.79
    #   2    15-0  30-0         ...       ...
    #   3    30-0  40-0         ...       ...
    #   4    40-0  NaN          ...       NaN
    df["V_before"] = df["Pts"].map(vDict)
    df["V_after"]  = df["next_state"].map(vDict)
    
    # For the last point of a game, next_state is missing (NaN), so V_after is also
    # missing (NaN). Put 1 to V_after if the server won the game, else 0. 
    df.loc[
        df["next_state"].isna(),
        "V_after"
    ] = df.loc[
        df["next_state"].isna(),
        "server_won_game"].astype(float)

    events = [classifyEvent(first, second) for first, second in zip(df["1st"].tolist(),
                                                                    df["2nd"].tolist())]
    df["event"]       = [e[0] if e else None for e in events]
    df["perspective"] = [e[1] if e else None for e in events]
    
    serverDeltaV = df["V_after"] - df["V_before"]
    multiplier = df["perspective"].map({
        "server":    1.0,
        "returner": -1.0,
    })
    df["delta_V"] = multiplier * serverDeltaV

    return df

# Compute the average delta_V for each event type. That average becomes the event's weight w. 
#   df: Point-by-point dataset that has been created by computeDeltaV().
#
def computeEventWeights(df: pd.DataFrame) -> pd.DataFrame:
    # Removes rows that have None/NaN in either/both of the "event" and "delta_V" columns
    valid = df.dropna(subset=["event", "delta_V"])
    
    # Groups rows according to their event types, selects delta_V values from each group,
    # and then calculate the average delta_V and the number of delta_V observations.
    # Output df:
    #   event           mean   count
    #   ace_or_winner   0.18   250
    #   double_fault    -0.16  140
    eventWeights = valid.groupby("event")["delta_V"].agg(["mean", "count"])
    
    eventWeights = eventWeights.rename(
        columns={"mean": "w", "count": "N"}
    ).reindex(EVENT_TYPES)
    
    wDict = eventWeights["w"].to_dict()
    
    return wDict, eventWeights


if __name__ == "__main__":
    from dataloader import MCPDataLoader
    from expectancy import computeGameWinExpectancy

    points = MCPDataLoader("w").points
    gweDict, gweDf, pts = computeGameWinExpectancy(points)
    gweDfSorted = gweDf.sort_values(["game win expectancy"])
    print(gweDfSorted)
    
    wDict, wDf = computeEventWeights( computeDeltaGameWinExpectancy(pts, gweDict) )
    print(wDf)
    print(wDict)
