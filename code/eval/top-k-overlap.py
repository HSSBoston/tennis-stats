import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from edge import EdgeCalc
from constants import OUTPUT_DIR
import pandas as pd
from scipy.stats import spearmanr

MIN_MATCHES = 15
TOP_EDGE_N = 20
TOP_WTA_N  = 20

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
    "Xin Yu Wang", # Xinyu Wang
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
    "Elena Gabriela Ruse", # Elena-Gabriela Ruse
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
calc = EdgeCalc(dl.points, dl.matches)

outputDict, outputDf = calc.playersEdge(players)
# print(f"{len(outputDict)} of {len(players)} players evaluated")

wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])                        
outputDf = pd.concat([outputDf, wtaRanks], axis=1)
outputDf = outputDf.dropna(subset=["player"])
# print(outputDf)

playersCountBefore = len(outputDf)
outputDf = outputDf[outputDf["matches"] >= MIN_MATCHES]
playersCountAfter = len(outputDf)
print(f"{len(players)-playersCountBefore} players excluded due to insufficient data")
print (f"{playersCountBefore - playersCountAfter} players excluded due to #matches<{MIN_MATCHES}")

numEligiblePlayers = len(outputDf)
print(f"\n{numEligiblePlayers} eligible players with #matches>={MIN_MATCHES}")
numEligibleInsideWtaTop50 = (
    outputDf.loc[:, "wta_rank"] <= TOP_WTA_N
).sum()
print(f"{numEligibleInsideWtaTop50} eligible players inside WTA top {TOP_WTA_N}")

outputDf = outputDf.sort_values("EDGE", ascending=False).reset_index(drop=True)
outputDf["edge_rank"] = outputDf.index + 1

edgeTop20Df = outputDf.loc[outputDf["edge_rank"] <= TOP_EDGE_N].copy()
numEdgeTop20InsideWtaTop50 = (
    edgeTop20Df.loc[:, "wta_rank"] <= TOP_WTA_N
).sum()
print(
    f"\nEDGE top {TOP_EDGE_N} players inside WTA top {TOP_WTA_N}: "
    f"{numEdgeTop20InsideWtaTop50}/{len(edgeTop20Df)}"
)

expectedRandomOverlap = (
    TOP_EDGE_N * numEligibleInsideWtaTop50 / numEligiblePlayers
)
print(
    f"Expected overlap under random selection from eligible players: "
    f"{expectedRandomOverlap:.2f}/{TOP_EDGE_N}"
)

print(f"\nEDGE top {TOP_EDGE_N} players outside WTA top {TOP_WTA_N}:")
highEdgeLowWtaDf = outputDf.loc[
    (outputDf["edge_rank"] <= TOP_EDGE_N) & (outputDf["wta_rank"] > TOP_WTA_N),
    ["edge_rank", "player", "EDGE", "wta_rank", "matches"]
]
print(highEdgeLowWtaDf)

print(f"\nWTA top {TOP_EDGE_N} players outside EDGE top {TOP_WTA_N}")
lowEdgeHighWtaDf = outputDf.loc[
    (outputDf["edge_rank"] > TOP_WTA_N) & (outputDf["wta_rank"] <= TOP_EDGE_N),
    ["edge_rank", "player", "EDGE", "wta_rank", "matches"]
]
print(lowEdgeHighWtaDf)

correlation, pValue = spearmanr(outputDf["edge_rank"], outputDf["wta_rank"])
print(f"\nSpearman correlation between EDGE rank and WTA rank: {correlation:.3f}")
print(f"p-value: {pValue:.4f}")

outputDf.to_csv(OUTPUT_DIR / "top-k-overlap.csv", index=False)
print(f"\nOutput written to: {OUTPUT_DIR}/top-k-overlap.csv")
