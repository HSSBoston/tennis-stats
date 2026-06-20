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
) -> tuple[float, dict] | None:
    # Extract rows where the Player 1 or Player 2 column equals playerName
    # Output with playerName = "Aryna Sabalenka":
    # match_id                                 Player 1          Player 2
    # 2026...-Aryna_Sabalenka-Victoria_Mboko   Aryna Sabalenka   Victoria Mboko
    # 2026...-Iga_Swiatek-Aryna_Sabalenka      Iga Swiatek       Aryna Sabalenka
    playerMatches= matches.loc[
        (matches["Player 1"] == playerName) | (matches["Player 2"] == playerName)
    ]
    if playerMatches.empty:
        return (None, None)

    # Output with playerName = "Aryna Sabalenka": 
    # { 2026...-Aryna_Sabalenka-Victoria_Mboko, 1,
    #   2026...-Iga_Swiatek-Aryna_Sabalenka,    2 }
    matchIdToPlayerNumber = {
        row["match_id"]: 1 if row["Player 1"] == playerName else 2
            for _, row in playerMatches.iterrows()
    }

    # Extract point-level rows where playerName played 
    df = df.loc[
        df["match_id"].isin(matchIdToPlayerNumber.keys())
    ]
    if df.empty:
        return (None, None)

    # Add "player_num" and "is_server" columns to the point-level data frame
    #   player_num (1 or 2) of playerName
    #   is_server (1 or 0): whether playerName serves (1) or not (0)
    # Output with playerName = "Aryna Sabalenka":
    #   match_id                                 Gm#   Pt   Svr   player_num   is_server
    #   2026...-Aryna_Sabalenka-Victoria_Mboko   1     1    1     1            1
    #   2026...-Iga_Swiatek-Aryna_Sabalenka      1     1    1     2            0
    df["player_num"] = df["match_id"].map(matchIdToPlayerNumber).astype("Int64")
    df["is_server"] = (df["player_num"] == df["Svr"])
    totalPoints = len(df)
    if totalPoints == 0:
        return (None, None)    

    # Removes rows that have None/NaN in at least one of the "svr", "event" and
    # "perspective" columns
    df = df.dropna(subset=["Svr", "event", "perspective"])
    classifiedPoints = len(df)
    if classifiedPoints == 0:
        return (None, None)
    
    # Extract point-level rows where
    #   (1) the event belongs to the server   and playerName serves, OR
    #   (2) the event belongs to the returner and playerName returns
    # These two cases (two types of points) are used to compute EDGE. In other words,
    # other cases are ignored; e.g., the event belongs to the server (e.g. the oponent's ace)
    # and playerName returns
    playerPts = df.loc[
        ( (df["perspective"] == "server")   &  df["is_server"]) |
        ( (df["perspective"] == "returner") & ~df["is_server"])
    ]
    attributedPoints = len(playerPts)
    if attributedPoints == 0:
        return (None, None)
    
    # Count each event type
    # value_counts() produces a data frame like:
    #   ace_or_winner  50
    #   double_fault   10
    #   unforced_error 20
    eventCountsDict = playerPts["event"].value_counts().to_dict()
    
    missingEvents = set(eventCountsDict) - set(wDict)
    if missingEvents:
        raise ValueError(f"Missing weights for event types: {sorted(missingEvents)}"
    )

    edgeNumerator = sum(
        wDict[event] * count
        for event, count in eventCountsDict.items()
    )
    edgePerTotalPoint      = edgeNumerator / totalPoints
    edgePerClassifiedPoint = edgeNumerator / classifiedPoints
    edgePerAttributedPoint = edgeNumerator / attributedPoints
    coverage  = classifiedPoints / totalPoints
    eventRate = attributedPoints / totalPoints
    
    return edgePerTotalPoint, {
        "player":    playerName,
        "EDGE":      edgePerTotalPoint,
        "coverage":  coverage,
        "EDGE2":     edgePerClassifiedPoint,
        "EDGE3":     edgePerAttributedPoint,
        "eventRate": eventRate,
        "points":    totalPoints,
        "matches":   len(playerMatches),
        "events":    eventCountsDict,
    }

if __name__ == "__main__":
    from dataloader import MCPDataLoader
    from winprob import computeV
    from eventweights import computeDeltaV, computeW
    from pprint import pprint

    dl = MCPDataLoader("w")
    points  = dl.points
    matches = dl.matches
    
    vDict, vDf, pointsV = computeV(points)
    vDfSorted = vDf.sort_values(["game win prob"])
    print(vDfSorted)
    
    pointsDeltaV = computeDeltaV(pointsV, vDict)
    wDict, wDf = computeW(pointsDeltaV)
    print(wDf)
#     print( wDict )

    edge, summary = computeEdge("Aryna Sabalenka", pointsDeltaV, matches, wDict)
    print(edge)
    pprint(summary)
    
