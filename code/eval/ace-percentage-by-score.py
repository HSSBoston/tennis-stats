"""Tabulate ace percentages using the same point sample as the EDGE analysis.

Run from any directory with Python and the project's pandas/numpy dependencies.
The default is the women's 2020s MCP file, with no additional date cutoff.
Edit TOUR and POINTS_FILES below to select another MCP sample.

Scores are pre-point scores, server first. Ace includes unreturnable serve
winners, as defined by eventparser.classifyEvent, on either serve. Each loaded
point counts once in the denominator, including unclassified/penalty points.
No deduplication or additional score filtering is applied beyond MCPDataLoader.
In particular, the loader retains tiebreak opening points labeled 0-0.
Absent score states have zero counts and a blank (undefined) percentage.
"""

import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(PRJ_DIR))

import pandas as pd
from constants import GAME_STATES, OUTPUT_DIR
from dataloader import MCPDataLoader
from eventparser import classifyEvent

TOUR = "w"
POINTS_FILES = None  # None uses charting-{TOUR}-points-2020s.csv


def computeAcePercentage(df: pd.DataFrame) -> pd.DataFrame:
    # Obtain df from MCPDataLoader.points. Keep all loaded points in the
    # denominator, rather than conditioning on a recognized event or a serve in.
    points = df[["Pts", "1st", "2nd"]].copy()
    if not points["Pts"].isin(GAME_STATES).all():
        raise ValueError("Expected only GAME_STATES; use MCPDataLoader.points")

    events = [classifyEvent(first, second) for first, second in zip(
        points["1st"].tolist(), points["2nd"].tolist())]
    points["is_ace"] = [event == ("ace", "server") for event in events]
    points["is_unclassified"] = [event is None for event in events]

    summaryDf = points.groupby("Pts").agg(
        total_server_points = ("is_ace", "size"),
        ace_count = ("is_ace", "sum"),
        unclassified_points = ("is_unclassified", "sum"),
    ).reindex(GAME_STATES, fill_value=0).astype(int)
    denominator = summaryDf["total_server_points"]
    summaryDf["ace_percentage"] = (
        100.0 * summaryDf["ace_count"] / denominator.where(denominator > 0)
    )
    summaryDf.index.name = "score_state"
    return summaryDf


if __name__ == "__main__":
    dl = MCPDataLoader(TOUR, pointsFiles=POINTS_FILES)
    summaryDf = computeAcePercentage(dl.points)

    print("\nAce includes unreturnable serve winners (EDGE convention).")
    print("Percentage = 100 * ace_count / total_server_points.")
    print("The loader's sample is preserved, including tiebreak openings at 0-0.")
    duplicateCount = dl.points.duplicated(["match_id", "Pt"]).sum()
    if duplicateCount:
        print(f"Warning: {duplicateCount} duplicate match/point rows retained.")
    unclassifiedCount = summaryDf["unclassified_points"].sum()
    print(f"{unclassifiedCount} unclassified points retained in denominators.")
    print()
    print(summaryDf.to_string(float_format=lambda value: f"{value:.4f}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputPath = OUTPUT_DIR / "ace-percentage-by-score.csv"
    summaryDf.to_csv(outputPath, float_format="%.6f")
    print("\nOutput written to:")
    print(outputPath)
