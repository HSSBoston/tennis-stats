import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from dr import DrCalc
from blr import BlrCalc
import pandas as pd

class DrPlusCalc:
    # points:  Point-level MCP data
    # matches: Match-level MCP data
    def __init__(self,
        points: pd.DataFrame,
        matches: pd.DataFrame
    ) -> None:
        self.points:  pd.DataFrame = points
        self.matches: pd.DataFrame = matches

        self.drCalc  = DrCalc(points, matches)
        self.blrCalc = BlrCalc(points, matches)

    # Compute DR+ for a given player
    def drPlus(self,
        playerName: str
    ) -> tuple[float | None, dict | None, pd.DataFrame | None]:
        
        drValue,  drSummaryDict,  _ = self.drCalc.dr(playerName)
        blrValue, blrSummaryDict, _ = self.blrCalc.blr(playerName)

        if drValue is None or blrValue is None:
            return (None, None, None)

        drPlusValue = drValue * blrValue

        summaryDict = {
            "player":  playerName,
            "DR+":     drPlusValue,
            "DR":      drValue,
            "BLR":     blrValue,

            "serve_LR":  blrSummaryDict["serve_LR"],
            "return_LR": blrSummaryDict["return_LR"],

            "serve_points_won_rate":  drSummaryDict["serve_points_won_rate"],
            "return_points_won_rate": drSummaryDict["return_points_won_rate"],
            "serve_points_lost_rate": drSummaryDict["serve_points_lost_rate"],

            "points":  drSummaryDict["points"],
            "matches": drSummaryDict["matches"],

            "serve_points":  drSummaryDict["serve_points"],
            "return_points": drSummaryDict["return_points"],

            "serve_points_won":   drSummaryDict["serve_points_won"],
            "return_points_won":  drSummaryDict["return_points_won"],

            "avg_serve_won_LEV":    blrSummaryDict["avg_serve_won_LEV"],
            "avg_serve_lost_LEV":   blrSummaryDict["avg_serve_lost_LEV"],
            "avg_return_won_LEV":   blrSummaryDict["avg_return_won_LEV"],
            "avg_return_lost_LEV":  blrSummaryDict["avg_return_lost_LEV"],
        }

        summaryDf = pd.DataFrame([summaryDict])
        return drPlusValue, summaryDict, summaryDf

    # Compute DR+ for multiple players
    def playersDrPlus(
        self,
        players: list[str]
    ) -> tuple[dict | None, pd.DataFrame | None]:

        outputDict = {}
        outputDf   = pd.DataFrame()

        for name in players:
            drPlusValue, _, summaryDf = self.drPlus(name)
            if drPlusValue is None:
                print(f"  {name}: not enough data — skipped.")
                continue

            outputDict[name] = drPlusValue
            outputDf = pd.concat([outputDf, summaryDf], axis=0, ignore_index=True)

        return outputDict, outputDf


if __name__ == "__main__":
    from pprint import pprint

    dl = MCPDataLoader("w")
    calc = DrPlusCalc(dl.points, dl.matches)

    drPlusValue, summaryDict, summaryDf = calc.drPlus("Aryna Sabalenka")
    print(drPlusValue)
    pprint(summaryDict)
    print(summaryDf)

    outputDict, outputDf = calc.playersDrPlus(
        ["Aryna Sabalenka", "Iga Swiatek", "Naomi Osaka"]
    )

    print(outputDict)
    print(outputDf)
