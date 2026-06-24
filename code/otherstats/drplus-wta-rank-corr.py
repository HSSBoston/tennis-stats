import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from drplus import DrPlusCalc
from constants import OUTPUT_DIR

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
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

dl = MCPDataLoader("w")
calc = DrPlusCalc(dl.points, dl.matches)
_, drPlusDf = calc.playersDrPlus(players)

wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])
df = pd.concat([drPlusDf, wtaRanks], axis=1)
df = df.dropna(subset=["player"])

playersCountBefore = len(df)
df = df.loc[ df["matches"] >= MIN_MATCHES ]
playersCountAfter = len(df)
print(f"{len(players)-playersCountBefore} players excluded due to insufficient data")
print(f"{playersCountBefore - playersCountAfter} players excluded due to #matches<{MIN_MATCHES}")

df = df.sort_values("DR+", ascending=False).reset_index(drop=True)
df["drplus_rank"] = df.index + 1
print(len(df), "players evaluated")

corr, pval = spearmanr(df["drplus_rank"], df["wta_rank"])

print(df[["player", "wta_rank", "DR+", "DR", "BLR",
          "drplus_rank", "points", "matches"]])
print()
print(f"Spearman correlation between DR+ rank and WTA rank: {corr:.3f}")
print(f"p-value: {pval:.6f}")

df.to_csv(OUTPUT_DIR / "drplus-wta-rank-correlation.csv", index=False)
print(f"\nOutput written to: {OUTPUT_DIR}/drplus-wta-rank-correlation.csv")

plt.figure(figsize=(8, 6))
plt.scatter(df["DR+"], df["wta_rank"])

# Trend line
slope, intercept = np.polyfit(df["DR+"], df["wta_rank"], 1)
xLine = np.linspace(df["DR+"].min(), df["DR+"].max(), 100)
yLine = slope * xLine + intercept

plt.plot(
    xLine, yLine, linestyle="--",
    linewidth=1.5, color="red", label="Linear trend")

# WTA rank #1 should appear near the top;
# set 0 exactly at the top edge of the y-axis.
maxWtaRank = df["wta_rank"].max()
plt.ylim(maxWtaRank + 5, 0)

plt.xlabel("Dominance Ratio Plus (DR+)")
plt.ylabel("WTA Rank (1 = best)")
plt.title(
    f"DR+ vs WTA Rank\n"
    f"WTA Top 100, matches >= {MIN_MATCHES}, n={len(df)}" )
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
