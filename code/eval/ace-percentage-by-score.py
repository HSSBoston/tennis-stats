# Generate a CSV file that records ace percentage at each score state. 
# Ace includes unreturnable serve winners, as defined by eventparser.classifyEvent.
#
# Each loaded point counts once in the denominator, including unclassified/penalty points.
#
# No deduplication or additional score filtering is applied beyond MCPDataLoader.
# In particular, the loader retains tiebreak opening points labeled 0-0.
#
# Absent score states have zero counts and a blank (undefined) percentage.

import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1]
sys.path.append(str(PRJ_DIR))

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from constants import GAME_STATES, OUTPUT_DIR
from dataloader import MCPDataLoader
from eventparser import classifyEvent


def computeAcePercentage(df: pd.DataFrame) -> pd.DataFrame:
    # Obtain df from MCPDataLoader.points. Keep all loaded points in the
    # denominator, rather than conditioning on a recognized event or a serve in.
    points = df[["Pts", "1st", "2nd"]].copy()
    if not points["Pts"].isin(GAME_STATES).all():
        raise ValueError("Unexpected/unrecognizable score state")

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
    dl = MCPDataLoader("w")
    summaryDf = computeAcePercentage(dl.points)

    duplicateCount = dl.points.duplicated(["match_id", "Pt"]).sum()
    if duplicateCount:
        print(f"Warning: {duplicateCount} duplicate match/point rows retained.")
        
    unclassifiedCount = summaryDf["unclassified_points"].sum()
    print(f"{unclassifiedCount} unclassified points retained in denominators.")
    print()
    print(summaryDf.to_string(float_format=lambda value: f"{value:.4f}"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputPath = OUTPUT_DIR / "ace-percentage-by-score.csv"
    summaryDf.to_csv(outputPath, float_format="%.2f")
    print("\nOutput written to:")
    print(outputPath)

    # Generate a heat map with a 5 x 5 grid to emphasize advantage states
    heatmap = np.full( (5, 5), np.nan )    

    scoreLabels = ["0", "15", "30", "40", "AD"]
    scoreToIndex = { score: i for i, score in enumerate(scoreLabels) }

    for scoreState, row in summaryDf.iterrows():
        serverScore, returnerScore = scoreState.split("-")
        if (serverScore in scoreToIndex and returnerScore in scoreToIndex):
            i = scoreToIndex[serverScore]
            j = scoreToIndex[returnerScore]
            heatmap[i, j] = row["ace_percentage"]

    # Mask invalid tennis score states so they remain blank.
    maskedHeatmap = np.ma.masked_invalid(heatmap)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(maskedHeatmap)

    ax.set_xticks(range(5))
    ax.set_xticklabels(scoreLabels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(scoreLabels)

    ax.set_xlabel("Returner Score")
    ax.set_ylabel("Server Score")
#     ax.set_title("Ace Percentage by Score State")

    # Add the percentage value to each valid cell.
    for i in range(5):
        for j in range(5):
            if not np.isnan(heatmap[i, j]):
                ax.text(j, i,
                        f"{heatmap[i, j]:.2f}%",
                        ha="center", va="center")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Ace Percentage (%)")

    fig.tight_layout()
    plt.show()

#     output_path = (
#         Path(__file__).resolve().parent.parent
#         / "output"
#         / "ace-percentage-by-score-heatmap.png"
#     )
# 
#     fig.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
# 
#     print(f"Saved heat map to: {output_path}")
