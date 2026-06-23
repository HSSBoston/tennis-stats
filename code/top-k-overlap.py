from dataloader import MCPDataLoader
from edge import EdgeCalc
from constants import OUTPUT_DIR
from pprint import pprint
import pandas as pd
from scipy.stats import spearmanr

# WTA top 100 players as of 06/15/2026
players = [ 
    "Aryna Sabalenka",
    "Elena Rybakina",
    "Iga Swiatek",
    "Jessica Pegula",
    "Mirra Andreeva",
    "Amanda Anisimova",
    "Coco Gauff",
    "Elina Svitolina",
    "Victoria Mboko",
    "Karolina Muchova",
    "Belinda Bencic",
    "Marta Kostyuk",
    "Linda Noskova",
    "Jasmine Paolini",
    "Naomi Osaka",
    "Diana Shnaider",
    "Iva Jovic",
    "Sorana Cirstea",
    "Ekaterina Alexandrova",
    "Anna Kalinskaya",
    "Maja Chwalinska",
    "Leylah Fernandez",
    "Clara Tauson",
    "Elise Mertens",
    "Emma Navarro",
    "Anastasia Potapova",
    "Marie Bouzkova",
    "Madison Keys",
    "Ann Li",
    "Hailey Baptiste",
    "Emma Raducanu",
    "Xinyu Wang",
    "Donna Vekic",
    "Katerina Siniakova",
    "Alexandra Eala",
    "Cristina Bucsa",
    "Liudmila Samsonova",
    "Jelena Ostapenko",
    "Barbora Krejcikova",
    "Jaqueline Cristian",
    "Maria Sakkari",
    "Laura Siegemund",
    "Janice Tjen",
    "McCartney Kessler",
    "Magdalena Frech",
    "Elisabetta Cocciaretto",
    "Sara Bejlek",
    "Magda Linette",
    "Marketa Vondrousova",
    "Dayana Yastremska",
    "Oleksandra Oliynykova",
    "Petra Marcinko",
    "Maya Joint",
    "Caty McNally",
    "Jessica Bouzas Maneiro",
    "Katie Boulter",
    "Antonia Ruzic",
    "Solana Sierra",
    "Yuliia Starodubtseva",
    "Diane Parry",
    "Zeynep Sonmez",
    "Nikola Bartunkova",
    "Tereza Valentova",
    "Peyton Stearns",
    "Kamilla Rakhimova",
    "Talia Gibson",
    "Shuai Zhang",
    "Panna Udvardy",
    "Daria Kasatkina",
    "Camila Osorio",
    "Anhelina Kalinina",
    "Varvara Gracheva",
    "Kimberly Birrell",
    "Anna Bondar",
    "Daria Snigur",
    "Viktorija Golubic",
    "Renata Zarazua",
    "Tamara Korpatsch",
    "Alycia Parks",
    "Eva Lys",
    "Taylor Townsend",
    "Elsa Jacquemot",
    "Sonay Kartal",
    "Lilli Tagger",
    "Yulia Putintseva",
    "Veronika Erjavec",
    "Karolina Pliskova",
    "Simona Waltert",
    "Oksana Selekhmeteva",
    "Anastasia Zakharova",
    "Maria Timofeeva",
    "Sinja Kraus",
    "Lanlana Tararudee",
    "Ella Seidel",
    "Ashlyn Krueger",
    "Ajla Tomljanovic",
    "Alina Korneeva",
    "Hanne Vandewinkel",
    "Francesca Jones",
    "Emiliana Arango",
]

MIN_MATCHES = 5
TOP_EDGE_N = 20
TOP_WTA_N = 20

dl = MCPDataLoader("w")
calc = EdgeCalc(dl.points, dl.matches)

outputDict, outputDf = calc.playersEdge(players)
print(f"{len(outputDict)} of {len(players)} players evaluated")

wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])                        
outputDf = pd.concat([outputDf, wtaRanks], axis=1)
outputDf = outputDf.dropna(subset=["player"])
print(outputDf)

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
print(f"\nOutput written to: {OUTPUT_DIR}/top-k-overlap .csv")
