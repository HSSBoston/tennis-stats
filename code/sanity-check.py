from dataloader import MCPDataLoader
from expectancy import computeGameWinExpectancy
from eventweights import computeDeltaGameWinExpectancy, computeEventWeights
from edge import computeEdge
from constants import OUTPUT_DIR
from pprint import pprint
import pandas as pd

# WTA ranking as of 06/15/2026
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

MIN_MATCHES = 14
TOP_EDGE_N = 20
TOP_WTA_N = 50

dl = MCPDataLoader("w")
points  = dl.points
matches = dl.matches

gweDict, gweDf, pointsGwe = computeGameWinExpectancy(points)
gweDfSorted = gweDf.sort_values(["game win expectancy"])
# print(gweDfSorted)

pointsDeltaGwe = computeDeltaGameWinExpectancy(pointsGwe, gweDict)
wDict, wDf = computeEventWeights(pointsDeltaGwe)
# print(wDf)
#     print( wDict )


rows = []
for index, name in enumerate(players):
    wtaRank = index + 1
    edge, summary = computeEdge(name, pointsDeltaGwe, matches, wDict)
    if edge is None:
        print(f"  {name}: not enough data — skipped.")
        continue
    summary["wta_rank"] = wtaRank
    rows.append(summary)
#     print(f"{name:<22} {edge:.5f}")

OUTPUT_DIR.mkdir(exist_ok=True)
gweDf.to_csv(OUTPUT_DIR / f"v-game-expectancy.csv")
wDf.to_csv(OUTPUT_DIR / f"w-event-weights.csv")

# Create a DataFrame from rows, while turning each row’s nested "events" dictionary
# into separate columns.
#   Example output:
#     player      EDGE   coverage   ace   double_faults ...
#     Sabalenka   0.33   0.99       207   145
outputDf = pd.DataFrame([
    {
        **{key: value   for key, value   in r.items() if key != "events"},
        **{event: count for event, count in r["events"].items()}
    }
    for r in rows
])

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

playersOutsideTop50 = edgeTop20Df.loc[
    edgeTop20Df["wta_rank"] > TOP_WTA_N,
    ["edge_rank", "player", "EDGE", "wta_rank", "matches"]
]

print("\nEDGE top 20 players outside WTA top 50:")
print(playersOutsideTop50)

outputDf.to_csv(OUTPUT_DIR / "sanity-check.csv", index=True)
print(f"\nOutput written to: {OUTPUT_DIR}")
