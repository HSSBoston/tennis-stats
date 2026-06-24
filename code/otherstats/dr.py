import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

import pandas as pd

class DrCalc:
    # points:  Point-level MCP data
    # matches: Match-level MCP data
    def __init__(self,
        points: pd.DataFrame,
        matches: pd.DataFrame
    ) -> None:
        self.points:  pd.DataFrame = points
        self.matches: pd.DataFrame = matches

    # Compute DR for a given player
    def dr(self,
        playerName: str
    ) -> tuple[float | None, dict | None, pd.DataFrame | None]:

        playerMatches = self.matches.loc[
            (self.matches["Player 1"] == playerName) | (self.matches["Player 2"] == playerName) ]
        if playerMatches.empty:
            return (None, None, None)

        matchIdToPlayerNumber = {
            row["match_id"]: 1 if row["Player 1"] == playerName else 2
            for _, row in playerMatches.iterrows()
        }

        df = self.points.copy()
        df = df.loc[ df["match_id"].isin(matchIdToPlayerNumber.keys()) ]

        if df.empty:
            return (None, None, None)
        df = df.dropna(subset=["Svr", "PtWinner"])
        if df.empty:
            return (None, None, None)

        df["player_num"] = df["match_id"].map(matchIdToPlayerNumber).astype("Int64")
        df["is_server"]  = (df["player_num"] == df["Svr"])
        df["won_point"]  = (df["player_num"] == df["PtWinner"])

        totalPoints = len(df)
        
        servePts  = df.loc[ df["is_server"]]
        returnPts = df.loc[~df["is_server"]]

        servePoints  = len(servePts)
        returnPoints = len(returnPts)
        if servePoints == 0 or returnPoints == 0:
            return (None, None, None)

        servePointsWon  = int(servePts["won_point"].sum())
        returnPointsWon = int(returnPts["won_point"].sum())

        servePointsWonRate  = servePointsWon / servePoints
        returnPointsWonRate = returnPointsWon / returnPoints
        servePointsLostRate = 1.0 - servePointsWonRate

        if servePointsLostRate == 0:
            drValue = float("inf")
        else:
            drValue = returnPointsWonRate / servePointsLostRate

        summaryDict = {
            "player": playerName,
            "DR":     drValue,
            "serve_points_won_rate":  servePointsWonRate,
            "return_points_won_rate": returnPointsWonRate,
            "serve_points_lost_rate": servePointsLostRate,
            "points":            totalPoints,
            "serve_points":      servePoints,
            "return_points":     returnPoints,
            "serve_points_won":  servePointsWon,
            "return_points_won": returnPointsWon,
            "matches":           len(playerMatches),
        }

        summaryDf = pd.DataFrame([summaryDict])

        return drValue, summaryDict, summaryDf

    # Compute DR for multiple players
    def playersDr(
        self,
        players: list[str]
    ) -> tuple[dict | None, pd.DataFrame | None]:

        outputDict = {}
        outputDf   = pd.DataFrame()

        for name in players:
            drValue, _, summaryDf = self.dr(name)
            if drValue is None:
                print(f"  {name}: not enough data — skipped.")
                continue
            outputDict[name] = drValue
            outputDf = pd.concat([outputDf, summaryDf], axis=0, ignore_index=True)

        return outputDict, outputDf


if __name__ == "__main__":
    from dataloader import MCPDataLoader
    from pprint import pprint

    dl = MCPDataLoader("w")
    calc = DrCalc(dl.points, dl.matches)

    drValue, summaryDict, summaryDf = calc.dr("Aryna Sabalenka")
    print(drValue)
    pprint(summaryDict)
    print(summaryDf)

    outputDict, outputDf = calc.playersDr(
        ["Aryna Sabalenka", "Iga Swiatek", "Naomi Osaka"]
    )

    print(outputDict)
    print(outputDf)