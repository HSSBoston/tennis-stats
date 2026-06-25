from dataloader import MCPDataLoader
from expectancy import computeGameWinExpectancy
from eventweights import computeDeltaGameWinExpectancy, computeEventWeights
from constants import OUTPUT_DIR
import pandas as pd

class EdgeCalc:
    # points:  Point-level MCP data (DataFrame). c.f. dataloader.MCPDataLoader.points
    # matches: Match-level MCP data (DataFrame). c.f. dataloader.MCPDataLoader.matches
    #
    def __init__(self,
        points: pd.DataFrame,
        matches: pd.DataFrame
    ) -> None:
        #  deltaGwePoints: original MCP DataFrame + extra columns "server_won_game",
        #    "next_state", "V_before", "V_after", "event", "perspective", "delta_V"
        #   wDict: Maps each event type to average game-win expectancy (delta_V)
        self.deltaGwePoints: pd.DataFrame
        self.wDict:          dict
        self.matches:        pd.DataFrame = matches
        
        gweDict, gweDf, pointsGwe = computeGameWinExpectancy(points)
        self.deltaGwePoints = computeDeltaGameWinExpectancy(pointsGwe, gweDict)
        self.wDict, wDf = computeEventWeights(self.deltaGwePoints)
        
        OUTPUT_DIR.mkdir(exist_ok=True)
        gweDf.to_csv(OUTPUT_DIR / "v-game-expectancy.csv")
        wDf.to_csv(  OUTPUT_DIR / "w-event-weights.csv")
        print("v-game-expectancy.csv and w-event-weights.csv saved")

    # Compute EDGE for a given player
    #   playerName: Player whose EDGE is being calculated. e.g. "Aryna Sabalenka"
    # Returns:
    #   edgePerTotalPoint: EDGE value
    #   summaryDict:
    #   summaryDf: 
    #
    def edge(self,
        playerName: str
    ) -> tuple[float | None, dict | None, pd.DataFrame | None]:       
        # Extract rows where the Player 1 or Player 2 column equals playerName
        # Output with playerName = "Aryna Sabalenka":
        #   match_id                                Player 1         Player 2
        #   2026...-Aryna_Sabalenka-Victoria_Mboko  Aryna Sabalenka  Victoria Mboko
        #   2026...-Iga_Swiatek-Aryna_Sabalenka     Iga Swiatek      Aryna Sabalenka
        playerMatches= self.matches.loc[
            (self.matches["Player 1"] == playerName) | (self.matches["Player 2"] == playerName)
        ]
        if playerMatches.empty:
            return (None, None, None)
        
        # Identify whether PlayerName is Player 1 or 2 in each match
        # Output with playerName = "Aryna Sabalenka": 
        #   { 2026...-Aryna_Sabalenka-Victoria_Mboko, 1,
        #     2026...-Iga_Swiatek-Aryna_Sabalenka,    2 }
        matchIdToPlayerNumber = {
            row["match_id"]: 1 if row["Player 1"] == playerName else 2
                for _, row in playerMatches.iterrows()
        }

        # Extract point-level rows where playerName played
        df = self.deltaGwePoints.copy()
        df = df.loc[
            df["match_id"].isin(matchIdToPlayerNumber.keys())
        ]
        if df.empty:
            return (None, None, None)

        # Add "player_num" and "is_server" columns to the point-level DataFrame
        #   player_num (1 or 2) of playerName
        #   is_server (1 or 0): whether playerName serves (1) or not (0)
        # Output with playerName = "Aryna Sabalenka":
        #   match_id                                Gm#  Pt  Svr  player_num  is_server
        #   2026...-Aryna_Sabalenka-Victoria_Mboko  1    1   1    1           1
        #   2026...-Iga_Swiatek-Aryna_Sabalenka     1    1   1    2           0
        df["player_num"] = df["match_id"].map(matchIdToPlayerNumber).astype("Int64")
        df["is_server"] = (df["player_num"] == df["Svr"])
        totalPoints = len(df)
        if totalPoints == 0:
            return (None, None, None)    

        # Remove rows that have None/NaN in at least one of the "svr", "event" and
        # "perspective" columns
        df = df.dropna(subset=["Svr", "event", "perspective"])
        classifiedPoints = len(df)
        if classifiedPoints == 0:
            return (None, None, None)
        
        # Extract point-level rows where
        #   (1) the event belongs to the server   and playerName serves, OR
        #   (2) the event belongs to the returner and playerName returns
        # These two cases (two types of points) are used to compute EDGE. In other words,
        # other cases are ignored; e.g., the event belongs to the server and playerName
        # returns (e.g. the oponent's ace)
        playerPts = df.loc[
            ( (df["perspective"] == "server")   & (df["is_server"] == 1) ) |
            ( (df["perspective"] == "returner") & (df["is_server"] == 0) )
        ]
        attributedPoints = len(playerPts)
        if attributedPoints == 0:
            return (None, None, None)
        
        # Count each event type
        # value_counts() produces a data frame like:
        #   ace_or_winner  50
        #   double_fault   10
        #   unforced_error 20
        eventCountsDict = playerPts["event"].value_counts().to_dict()
        
        missingEvents = set(eventCountsDict) - set(self.wDict)
        if missingEvents:
            raise ValueError(f"Missing weights for event types: {sorted(missingEvents)}"
        )

        positiveNumerator = 0.0
        negativeNumerator = 0.0
        eventEdgeDict = {}

        for event, count in eventCountsDict.items():
            contribution = self.wDict[event] * count
            eventEdgeDict[event] = contribution / totalPoints
            if contribution > 0:
                positiveNumerator += contribution
            elif contribution < 0:
                negativeNumerator += contribution

        edgeNumerator = positiveNumerator + negativeNumerator
        edgePerTotalPoint      = edgeNumerator / totalPoints
    #     edgePerClassifiedPoint = edgeNumerator / classifiedPoints
    #     edgePerAttributedPoint = edgeNumerator / attributedPoints

        positiveEdge = positiveNumerator / totalPoints
        negativeEdge = negativeNumerator / totalPoints

        coverage  = classifiedPoints / totalPoints
        eventRate = attributedPoints / totalPoints

        summaryDic = {
            "player":             playerName,
            "EDGE":               edgePerTotalPoint,
            "positive_EDGE":      positiveEdge,
            "negative_EDGE":      negativeEdge,
            "event_EDGE_contrib": eventEdgeDict,
    #         "EDGE2":        edgePerClassifiedPoint,
    #         "EDGE3":        edgePerAttributedPoint,
            "points":       totalPoints,
            "matches":      len(playerMatches),
            "coverage":     coverage,
            "eventRate":    eventRate,
            "event_counts": eventCountsDict,
        }

        # Create a DataFrame from summaryDic, turning/flattening summaryDic's
        # nested dictionaries into separate columns
        #   Example output:
        #     player      EDGE   coverage  ace_edge  double_faults_edge ...
        #     Sabalenka   0.33   0.99      207       145
        summaryDf = pd.DataFrame([
            {
                **{key: value for key, value in summaryDic.items()
                   if (key != "event_counts") and (key != "event_EDGE_contrib")},
                **{ev+"_edge": evEdge
                   for ev, evEdge in summaryDic["event_EDGE_contrib"].items()},
                **{ev+"_count": count
                   for ev, count in summaryDic["event_counts"].items()}
            }
        ])

        return edgePerTotalPoint, summaryDic, summaryDf

    # Compute EDGE for multiple players
    #   players: List of player names e.g. ["Aryna Sabalenka", , "Iga Swiatek"]
    # Returns:
    #   outputDict:
    #   outputDf: 
    #
    def playersEdge(self,
        players: list[str]
    ) -> tuple[ dict | None, pd.DataFrame | None]:
        
        outputDict = {}
        outputDf   = pd.DataFrame()
        
        for name in players:
            edge, _, summaryDf = self.edge(name)
            if edge is None:
                print(f"  {name}: not enough data — skipped.")
                continue
            outputDict[name] = edge
            outputDf = pd.concat([outputDf, summaryDf], axis=0, ignore_index=True)
        
        return outputDict, outputDf
            
if __name__ == "__main__":
    from pprint import pprint

    dl = MCPDataLoader("w")
    calc = EdgeCalc(dl.points, dl.matches)
    
    edge, summaryDic, summaryDf = calc.edge("Aryna Sabalenka")
    print(edge)
    pprint(summaryDic)
    print(summaryDf)
    
    outputDict, outputDf = calc.playersEdge(
        ["Aryna Sabalenka",
         "Iga Swiatek",
         "Naomi Osaka"])
    print(outputDict)
    print(outputDf)
    
