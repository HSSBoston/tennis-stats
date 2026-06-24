import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from blr import BlrCalc
from constants import OUTPUT_DIR

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

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

dl = MCPDataLoader("w")
calc = BlrCalc(dl.points, dl.matches)
_, blrDf = calc.playersBlr(players)

# Add WTA ranks based on player-list order
wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])                        
df = pd.concat([blrDf, wtaRanks], axis=1)
df = df.dropna(subset=["player", "BLR", "wta_rank"])

playersCountBefore = len(df)
df = df.loc[ df["matches"] >= MIN_MATCHES ]
playersCountAfter = len(df)
print(f"{len(players)-playersCountBefore} players excluded due to insufficient data")
print (f"{playersCountBefore - playersCountAfter} players excluded due to #matches<{MIN_MATCHES}")

df = df.sort_values("BLR", ascending=False).reset_index(drop=True)
df["blr_rank"] = df.index + 1
print(len(df), "players evaluated")

# Spearman correlation: BLR rank vs WTA rank
corr, pValue = spearmanr(df["blr_rank"], df["wta_rank"])

print(f"Spearman correlation between BLR rank and WTA rank: {corr:.3f}")
print(f"p-value: {pValue:.4f}")
print(df[["player", "BLR", "blr_rank", "wta_rank", "matches", "points"]])

OUTPUT_DIR.mkdir(exist_ok=True)
df.to_csv(OUTPUT_DIR / "blr-wta-rank-correlation.csv", index=False)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df["BLR"], df["wta_rank"])

# Trend line
slope, intercept = np.polyfit(df["BLR"], df["wta_rank"], 1)
xLine = np.linspace(df["BLR"].min(), df["BLR"].max(), 100)
yLine = slope * xLine + intercept

plt.plot(
    xLine, yLine, linestyle="--",
    linewidth=1.5, color ="red", label="Linear trend")

# WTA rank #1 should appear near the top;
# set 0 exactly at the top edge of the y-axis.
maxWtaRank = df["wta_rank"].max()
plt.ylim(maxWtaRank + 5, 0)

plt.xlabel("Balanced Leverage Ratio (BLR)")
plt.ylabel("WTA Rank (1 = best)")
plt.title(
    f"Balanced Leverage Ratio (BLR) vs WTA Rank\n"
    f"WTA Top 100, matches >= {MIN_MATCHES}, n={len(df)}" )

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


