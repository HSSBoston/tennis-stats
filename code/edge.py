# EDGE - Event-Driven Gain in Expectancy

import pandas as pd

# Compute the EDGE value for a given player
#   playerName: Player whose EDGE is being calculated. e.g. "Aryna Sabalenka"
#   df: Point-level data that has been created by eventweights.computeDeltaV()
#   matches: Match-level data. c.f. dataloader.MCPDataLoader.matches
#   wDict: Maps each event type to its weight
#
def computeEdge(
    playerName: str,
    df: pd.DataFrame,
    matches: pd.DataFrame,
    wDict: dict
) -> dict | None:
    # Extract rows where the Player 1 or Player 2 column equals playerName
    # Output with playerName = "Aryna Sabalenka":
    # match_id                                 Player 1          Player 2
    # 2026...-Aryna_Sabalenka-Victoria_Mboko   Aryna Sabalenka   Victoria Mboko
    # 2026...-Iga_Swiatek-Aryna_Sabalenka      Iga Swiatek       Aryna Sabalenka
    playerMatches= matches.loc[
        (matches["Player 1"] == playerName) | (matches["Player 2"] == playerName)
    ]
    if playerMatches.empty:
        return None

    # Output with playerName = "Aryna Sabalenka": 
    # { 2026...-Aryna_Sabalenka-Victoria_Mboko, 1,
    #   2026...-Iga_Swiatek-Aryna_Sabalenka,    2 }
    matchIdToPlayerNumber = {
        row["match_id"]: 1 if row["Player 1"] == playerName else 2
            for _, row in playerMatches.iterrows()
    }

    df = df.loc[
        df["match_id"].isin(matchIdToPlayerNumber.keys())
    ]
    if df.empty:
        return None
    
    df["player_num"] = df["match_id"].map(matchIdToPlayerNumber).astype("Int64")
    df["is_server"] = (df["player_num"] == df["Svr"])

    df = df.dropna(subset=["event", "perspective"])

    playerPts = df.loc[
        ( (df["perspective"] == "server")   &  df["is_server"]) |
        ( (df["perspective"] == "returner") & ~df["is_server"])
    ]
    N = len(playerPts)
    if N == 0:
        return None
    
    counts = playerPts["event"].value_counts().to_dict()
    X = sum(wDict.get(e, 0.0) * c for e, c in counts.items()) / N
    
    return {"player": playerName, "X": X, "N": N, "counts": counts,
            "matches": int(playerMatches.shape[0])}



if __name__ == "__main__":
    from constants import playersW, playersM
    from dataloader import MCPDataLoader
    from winprob import computeV
    from eventweights import computeDeltaV, computeW

    dl = MCPDataLoader("w", playersW)
    points  = dl.points
    matches = dl.matches
    
    vDict, vDf, pointsV = computeV(points)
    vDfSorted = vDf.sort_values(["game win prob"])
    print(vDfSorted)
    
    pointsDeltaV = computeDeltaV(pointsV, vDict)
    wDict, wDf = computeW(pointsDeltaV)
    print(wDf)
#     print( wDict )

    playerEdge = computeEdge("Aryna Sabalenka", pointsDeltaV, matches, wDict)
    print(playerEdge)
    
