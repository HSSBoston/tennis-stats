import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from edge import EdgeCalc
from constants import OUTPUT_DIR
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np

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

MIN_MATCHES = 10
EDGE_SCALE = 1000

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

outputDf = outputDf.sort_values("EDGE", ascending=False).reset_index(drop=True)
outputDf["edge_rank"] = outputDf.index + 1
print(len(outputDf), "players evaluated")

correlation, pValue = spearmanr(outputDf["edge_rank"], outputDf["wta_rank"])
print(f"\nSpearman correlation between EDGE rank and WTA rank: {correlation:.3f}")
print(f"p-value: {pValue:.4f}")

outputDf["scaled_EDGE"] = outputDf["EDGE"] * EDGE_SCALE
outputDf.to_csv(OUTPUT_DIR / "edge-wta-rank-corr.csv", index=False)
print(f"\nOutput written to: {OUTPUT_DIR}/edge-wta-rank-corr.csv")

plt.figure(figsize=(8, 6))
plt.scatter( outputDf["scaled_EDGE"], outputDf["wta_rank"] )

# Trend line
slope, intercept = np.polyfit(outputDf["scaled_EDGE"], outputDf["wta_rank"], 1)
xLine = np.linspace(outputDf["scaled_EDGE"].min(), outputDf["scaled_EDGE"].max(), 100)
yLine = slope * xLine + intercept

plt.plot(
    xLine, yLine, linestyle="--",
    linewidth=1.5, color ="red", label="Linear trend")

# WTA rank #1 should appear near the top;
# set 0 exactly at the top edge of the y-axis.
maxWtaRank = outputDf["wta_rank"].max()
plt.ylim(maxWtaRank + 5, 0)

plt.title(
    f"EDGE Value vs WTA Rank\n"
    f"WTA Top 100, matches >= {MIN_MATCHES}, n={len(outputDf)}")
plt.xlabel("Scaled EDGE value (EDGE × 1000)")
plt.ylabel("WTA rank (1 = best)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

