import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from expectancy import computeGameWinExpectancy
from eventweights import computeDeltaGameWinExpectancy
from constants import OUTPUT_DIR
import pandas as pd


class BlrCalc:
    # points:  Point-level MCP data (DataFrame). c.f. dataloader.MCPDataLoader.points
    # matches: Match-level MCP data (DataFrame). c.f. dataloader.MCPDataLoader.matches
    #
    def __init__(self,
        points: pd.DataFrame,
        matches: pd.DataFrame
    ) -> None:
        # deltaGwePoints: original MCP data + extra columns "server_won_game",
        # "next_state", "V_before", "V_after", "event", "perspective", "delta_V"
        self.deltaGwePoints: pd.DataFrame
        self.matches: pd.DataFrame = matches

        gweDict, gweDf, pointsGwe = computeGameWinExpectancy(points)
        self.deltaGwePoints = computeDeltaGameWinExpectancy(pointsGwe, gweDict)

        OUTPUT_DIR.mkdir(exist_ok=True)
        gweDf.to_csv(OUTPUT_DIR / "v-game-expectancy.csv")
        print("v-game-expectancy.csv saved")

    # Compute leverage ratio (LR)
    #   wonLev:  LEV values of points won by the player
    #   lostLev: LEV values of points lost by the player
    # Returns:
    #   ratio: average LEV of won points / average LEV of lost points
    #
    def _leverageRatio(self,
        wonLev: pd.Series,
        lostLev: pd.Series
    ) -> float | None:
        if len(wonLev) == 0 or len(lostLev) == 0:
            return None

        avgWonLev  = wonLev.mean()
        avgLostLev = lostLev.mean()

        if pd.isna(avgWonLev) or pd.isna(avgLostLev) or avgLostLev == 0:
            return None

        return avgWonLev / avgLostLev

    # Compute BLR for a given player
    #   playerName: Player whose BLR is being calculated. e.g. "Aryna Sabalenka"
    # Returns:
    #   blr: BLR
    #   summaryDict:
    #   summaryDf:
    #
    def blr(self,
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

        # Extract point-level rows where playerName played
        df = self.deltaGwePoints.copy()
        df = df.loc[
            df["match_id"].isin(matchIdToPlayerNumber.keys())
        ]
        if df.empty:
            return (None, None, None)

        # Add player_num, is_server, and won_point columns
        df["player_num"] = df["match_id"].map(matchIdToPlayerNumber).astype("Int64")
        df["is_server"]  = (df["player_num"] == df["Svr"])
        df["won_point"]  = (df["player_num"] == df["PtWinner"])

        # Keep rows with enough information
        df = df.dropna(subset=["Svr", "PtWinner", "delta_V"])
        totalPoints = len(df)
        if totalPoints == 0:
            return (None, None, None)

        # For BLR, leverage (LEV) is the magnitude of game win expectancy (delta_V)
        df["LEV"] = df["delta_V"].abs()

        servePts  = df.loc[df["is_server"] == 1]
        returnPts = df.loc[df["is_server"] == 0]

        serveWonLev  = servePts.loc[servePts["won_point"] == 1, "LEV"]
        serveLostLev = servePts.loc[servePts["won_point"] == 0, "LEV"]

        returnWonLev  = returnPts.loc[returnPts["won_point"] == 1, "LEV"]
        returnLostLev = returnPts.loc[returnPts["won_point"] == 0, "LEV"]

        serveLR  = self._leverageRatio(serveWonLev, serveLostLev)
        returnLR = self._leverageRatio(returnWonLev, returnLostLev)

        if serveLR is None and returnLR is None:
            return (None, None, None)
        elif serveLR is None:
            blr = returnLR
        elif returnLR is None:
            blr = serveLR
        else:
            blr = (serveLR + returnLR) / 2

        summaryDic = {
            "player": playerName,
            "BLR": blr,
            "serve_LR": serveLR,
            "return_LR": returnLR,

            "points": totalPoints,
            "matches": len(playerMatches),

            "serve_points": len(servePts),
            "return_points": len(returnPts),

            "serve_points_won": len(serveWonLev),
            "serve_points_lost": len(serveLostLev),
            "return_points_won": len(returnWonLev),
            "return_points_lost": len(returnLostLev),

            "avg_serve_won_LEV": serveWonLev.mean(),
            "avg_serve_lost_LEV": serveLostLev.mean(),
            "avg_return_won_LEV": returnWonLev.mean(),
            "avg_return_lost_LEV": returnLostLev.mean(),
        }

        summaryDf = pd.DataFrame([summaryDic])

        return blr, summaryDic, summaryDf

    # Compute BLR for multiple players
    #   players: List of player names e.g. ["Aryna Sabalenka", "Iga Swiatek"]
    # Returns:
    #   outputDict:
    #   outputDf:
    #
    def playersBlr(self,
        players: list[str]
    ) -> tuple[dict | None, pd.DataFrame | None]:

        outputDict = {}
        outputDf   = pd.DataFrame()

        for name in players:
            blr, _, summaryDf = self.blr(name)
            if blr is None:
                print(f"  {name}: not enough data — skipped.")
                continue

            outputDict[name] = blr
            outputDf = pd.concat([outputDf, summaryDf], axis=0, ignore_index=True)

        return outputDict, outputDf


if __name__ == "__main__":
    from pprint import pprint

    dl = MCPDataLoader("w")
    calc = BlrCalc(dl.points, dl.matches)

    blr, summaryDic, summaryDf = calc.blr("Aryna Sabalenka")
    print(blr)
    pprint(summaryDic)
    print(summaryDf)

    outputDict, outputDf = calc.playersBlr(
        ["Aryna Sabalenka",
         "Iga Swiatek",
         "Naomi Osaka"]
    )
    print(outputDict)
    print(outputDf)

