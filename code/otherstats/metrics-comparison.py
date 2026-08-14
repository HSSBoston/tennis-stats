import pathlib
import sys
PRJ_DIR = pathlib.Path(__file__).parents[1]  # 2 levels up
sys.path.append(str(PRJ_DIR))

import pandas as pd
from dataloader import MCPDataLoader
from edge import EdgeCalc
from dr import DrCalc
from drplus import DrPlusCalc
from constants import OUTPUT_DIR

MIN_MATCHES = 5
WTA_RANKING_DATE = pd.Timestamp("2026-05-25")
ANALYSIS_START_DATE = WTA_RANKING_DATE - pd.DateOffset(years=2)
ANALYSIS_END_DATE = WTA_RANKING_DATE - pd.Timedelta(days=1)

# WTA top 100 players as of 05/25/2026
players = [
    "Aryna Sabalenka",
    "Elena Rybakina",
    "Iga Swiatek",
    "Coco Gauff",
    "Jessica Pegula",
    "Amanda Anisimova",
    "Elina Svitolina",
    "Mirra Andreeva",
    "Victoria Mboko",
    "Karolina Muchova",
    "Belinda Bencic",
    "Linda Noskova",
    "Jasmine Paolini",
    "Ekaterina Alexandrova",
    "Marta Kostyuk",
    "Naomi Osaka",
    "Iva Jovic",
    "Sorana Cirstea",
    "Madison Keys",
    "Clara Tauson",
    "Elise Mertens",
    "Leylah Fernandez",
    "Diana Shnaider",
    "Anna Kalinskaya",
    "Emma Navarro",
    "Hailey Baptiste",
    "Liudmila Samsonova",
    "Marie Bouzkova",
    "Ann Li",
    "Anastasia Potapova",
    "Jelena Ostapenko",
    "Jaqueline Cristian",
    "Cristina Bucsa",
    "Xin Yu Wang",  # Xinyu Wang
    "Sara Bejlek",
    "Katerina Siniakova",
    "Alexandra Eala",
    "Elisabetta Cocciaretto",
    "Emma Raducanu",
    "Janice Tjen",
    "Barbora Krejcikova",
    "Tereza Valentova",
    "Lois Boisson",
    "Marketa Vondrousova",
    "Dayana Yastremska",
    "Magdalena Frech",
    "Laura Siegemund",
    "Mccartney Kessler",
    "Maria Sakkari",
    "Jessica Bouzas Maneiro",
    "Petra Marcinko",
    "Maya Joint",
    "Daria Kasatkina",
    "Tatjana Maria",
    "Yuliia Starodubtseva",
    "Qinwen Zheng",
    "Anna Bondar",
    "Talia Gibson",
    "Panna Udvardy",
    "Anhelina Kalinina",
    "Shuai Zhang",
    "Sonay Kartal",
    "Caty Mcnally",
    "Antonia Ruzic",
    "Oleksandra Oliynykova",
    "Zeynep Sonmez",
    "Elsa Jacquemot",
    "Solana Sierra",
    "Nikola Bartunkova",
    "Varvara Gracheva",
    "Katie Boulter",
    "Donna Vekic",
    "Magda Linette",
    "Renata Zarazua",
    "Taylor Townsend",
    "Yulia Putintseva",
    "Elena Gabriela Ruse",  # Elena-Gabriela Ruse
    "Peyton Stearns",
    "Alycia Parks",
    "Anastasia Zakharova",
    "Eva Lys",
    "Viktorija Golubic",
    "Kimberly Birrell",
    "Veronika Erjavec",
    "Veronika Kudermetova",
    "Camila Osorio",
    "Sofia Kenin",
    "Oksana Selekhmeteva",
    "Kamilla Rakhimova",
    "Lilli Tagger",
    "Simona Waltert",
    "Diane Parry",
    "Daria Snigur",
    "Emiliana Arango",
    "Tamara Korpatsch",
    "Ella Seidel",
    "Lanlana Tararudee",
    "Sinja Kraus",
    "Hanne Vandewinkel",
    "Ajla Tomljanovic"
]

dl = MCPDataLoader("w")

# Restrict both match metadata and point data to the same rolling
# two-year window preceding the WTA ranking date.
loadedPointMatchIds = set(
    dl.points["match_id"].dropna().unique()
)

analysisMatches = dl.matches.loc[
    dl.matches["match_id"].isin(loadedPointMatchIds)
].copy()
analysisMatches["match_date"] = pd.to_datetime(
    analysisMatches["Date"],
    format="%Y%m%d",
    errors="raise"
)

analysisMatches = analysisMatches.loc[
    analysisMatches["match_date"].between(
        ANALYSIS_START_DATE,
        ANALYSIS_END_DATE,
        inclusive="both"
    )
].copy()

analysisMatchIds = set(
    analysisMatches["match_id"].dropna().unique()
)

analysisPoints = dl.points.loc[
    dl.points["match_id"].isin(analysisMatchIds)
].copy()

# Remove metadata for matches that do not have point data in the
# loaded MCP point file.
pointMatchIds = set(
    analysisPoints["match_id"].dropna().unique()
)

analysisMatches = analysisMatches.loc[
    analysisMatches["match_id"].isin(pointMatchIds)
].copy()

if analysisPoints.empty or analysisMatches.empty:
    raise ValueError("No MCP data found within the analysis window")

rankDf = pd.DataFrame( {
    "player":   players,
    "WTA_rank": range(1, len(players) + 1)
} )

# Record the first and last charted matches actually used for each
# player within the common analysis window.
playerDatesDf = pd.concat(
    [
        analysisMatches[
            ["Player 1", "match_date"]
        ].rename(columns={"Player 1": "player"}),
        analysisMatches[
            ["Player 2", "match_date"]
        ].rename(columns={"Player 2": "player"}),
    ],
    ignore_index=True
)

playerDatesDf = (
    playerDatesDf
    .groupby("player", as_index=False)["match_date"]
    .agg(
        FIRST_MATCH_DATE="min",
        LAST_MATCH_DATE="max"
    )
)

for dateColumn in ["FIRST_MATCH_DATE", "LAST_MATCH_DATE"]:
    playerDatesDf[dateColumn] = (
        playerDatesDf[dateColumn].dt.strftime("%Y-%m-%d")
    )

# Calculate EDGE

_, edgeDf = EdgeCalc(
    analysisPoints,
    analysisMatches
).playersEdge(players)
edgeDf = edgeDf[ ["player", "EDGE", "matches"] ].copy()
edgeDf = edgeDf.rename( columns={"matches": "EDGE_matches"} )


# Calculate DR

_, drDf = DrCalc(
    analysisPoints,
    analysisMatches
).playersDr(players)
drDf = drDf[ ["player", "DR", "matches"] ].copy()
drDf = drDf.rename( columns={"matches": "DR_matches"} )

# Calculate DR+

_, drPlusDf = DrPlusCalc(
    analysisPoints,
    analysisMatches
).playersDrPlus(players)
drPlusDf = drPlusDf[ ["player", "DR+", "matches"] ].copy()
drPlusDf = drPlusDf.rename( columns={"matches": "DRPlus_matches"} )

resultDf = (
    rankDf
    .merge(playerDatesDf, on="player", how="left")
    .merge(edgeDf,   on="player", how="left")
    .merge(drDf,     on="player", how="left")
    .merge(drPlusDf, on="player", how="left") )

# Verify "EDGE_matches", "DR_matches", and "DRPlus_matches" have the same number
inconsistentPlayers = resultDf.loc[
    resultDf[
        ["EDGE_matches", "DR_matches", "DRPlus_matches"]
    ].nunique(axis=1, dropna=False) != 1,
    ["player", "EDGE_matches", "DR_matches", "DRPlus_matches"],
]

if not inconsistentPlayers.empty:
    raise ValueError( f"Match counts differ:\n{inconsistentPlayers}")

resultDf["matches"] = (
    resultDf["EDGE_matches"].astype("Int64")
)

# Rank each metric from highest (best) to lowest. Nullable integers
# preserve missing metric values as blank cells in the output CSV.
resultDf["EDGE_rank"] = resultDf["EDGE"].rank(
    ascending=False,
    method="min",
    na_option="keep"
).astype("Int64")

resultDf["DR_rank"] = resultDf["DR"].rank(
    ascending=False,
    method="min",
    na_option="keep"
).astype("Int64")

resultDf["DRPlus_rank"] = resultDf["DR+"].rank(
    ascending=False,
    method="min",
    na_option="keep"
).astype("Int64")

#print(resultDf)

eligibleDf = resultDf[
    resultDf["matches"] >= MIN_MATCHES
].copy()

eligibleDf["WTA_eligible_rank"] = eligibleDf["WTA_rank"].rank(
    ascending=True,
    method="min",
    na_option="keep"
).astype("Int64")

rankColumns = {
    "EDGE": "EDGE_rank",
    "DR":   "DR_rank",
    "DR+":  "DRPlus_rank" }

for metricColumn, rankColumn in rankColumns.items():
    eligibleDf[rankColumn] = eligibleDf[metricColumn].rank(
        ascending=False,
        method="min",
        na_option="keep"
    ).astype("Int64")

eligibleDf = eligibleDf[[
    "player",
    "WTA_rank",
    "WTA_eligible_rank",
    "matches",
    "FIRST_MATCH_DATE",
    "LAST_MATCH_DATE",
    "EDGE",
    "EDGE_rank",
    "DR",
    "DR_rank",
    "DR+",
    "DRPlus_rank",
]]

print(
    "Analysis window: "
    f"{ANALYSIS_START_DATE:%Y-%m-%d} through "
    f"{ANALYSIS_END_DATE:%Y-%m-%d}"
)
print(f"WTA top 100 players: {len(resultDf)}")
print(f"Players with >= {MIN_MATCHES} matches: {len(eligibleDf)}")

outputFile = "edge-dr-drplus-wta-top100-all.csv"
resultDf.to_csv(
    OUTPUT_DIR / outputFile,
    index=False
)
print(f"\nSaved to: {outputFile}")

outputFile = f"edge-dr-drplus-wta-top100-min{MIN_MATCHES}.csv"
eligibleDf.to_csv(
    OUTPUT_DIR / outputFile,
    index=False
)
print(f"Saved to: {outputFile}")
