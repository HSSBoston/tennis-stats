import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from edge import EdgeCalc
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

MIN_MATCHES = 15
EDGE_SCALE = 1000

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

_, outputDf = calc.playersEdge(players)
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
print(f"{playersCountAfter} players included in histogram")
print()

scaledEdgeValues = outputDf["EDGE"] * EDGE_SCALE
meanEdge   = scaledEdgeValues.mean()
medianEdge = scaledEdgeValues.median()
stdEdge    = scaledEdgeValues.std()
minEdge    = scaledEdgeValues.min()
maxEdge    = scaledEdgeValues.max()
print(f"Mean:    {meanEdge:.2f}")
print(f"Median:  {medianEdge:.2f}")
print(f"Std dev: {stdEdge:.2f}")
print(f"Min:     {minEdge:.2f}")
print(f"Max:     {maxEdge:.2f}")
print(f"Range:   {maxEdge-minEdge:.2f}")


outputDf["scaled_EDGE"] = scaledEdgeValues
q1  = outputDf["scaled_EDGE"].quantile(0.25)
q3  = outputDf["scaled_EDGE"].quantile(0.75)
iqr = q3 - q1
lowerBound = q1 - 1.5 * iqr
upperBound = q3 + 1.5 * iqr

outlierDf = outputDf[
    (outputDf["scaled_EDGE"] < lowerBound) | (outputDf["scaled_EDGE"] > upperBound) ]
print()
print("IQR outlier check")
print(f"Q1:          {q1:.2f}")
print(f"Q3:          {q3:.2f}")
print(f"IQR:         {iqr:.2f}")
print(f"Lower bound: {lowerBound:.2f}")
print(f"Upper bound: {upperBound:.2f}")
print(f"# outliers:  {len(outlierDf)}")

if len(outlierDf) > 0:
    print()
    print("Outlier players:")
    print(
        outlierDf[["player", "wta_rank", "matches", "scaled_EDGE"]]
        .sort_values("scaled_EDGE", ascending=False)
        .to_string(index=False)
    )
else:
    print("No IQR-based outliers found")

plt.figure(figsize=(8, 5))
plt.hist(scaledEdgeValues, bins=10, edgecolor="black")

plt.axvline(
    meanEdge, linestyle="-", linewidth=1.5,
    label=f"Mean: {meanEdge:.2f}")

plt.axvline(
    medianEdge, linestyle=":", linewidth=1.5,
    label=f"Median: {medianEdge:.2f}" )

plt.axvspan(
    q1, q3, alpha=0.15,
    label=f"IQR: {q1:.2f}–{q3:.2f}" )

plt.axvline(
    lowerBound, linestyle="--", linewidth=1.5,
    label=f"Lower bound: {lowerBound:.2f}" )

plt.axvline(
    upperBound, linestyle="--", linewidth=1.5,
    label=f"Upper bound: {upperBound:.2f}" )

# plt.title(
#     f"Distribution of Player EDGE Values\n"
#     f"WTA Top 100, matches >= {MIN_MATCHES}, n={len(scaledEdgeValues)}" )
plt.xlabel("Scaled EDGE value (EDGE × 1000)", fontsize=16)
plt.ylabel("Number of players", fontsize=16)
plt.tick_params(axis="both", labelsize=14)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()

