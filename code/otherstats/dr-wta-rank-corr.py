import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from dr import DrCalc
from constants import OUTPUT_DIR

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np

MIN_MATCHES = 15

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
calc = DrCalc(dl.points, dl.matches)
_, drDf = calc.playersDr(players)

wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])                        
df = pd.concat([drDf, wtaRanks], axis=1)
df = df.dropna(subset=["player"])

playersCountBefore = len(df)
df = df.loc[ df["matches"] >= MIN_MATCHES ]
playersCountAfter = len(df)
print(f"{len(players)-playersCountBefore} players excluded due to insufficient data")
print (f"{playersCountBefore - playersCountAfter} players excluded due to #matches<{MIN_MATCHES}")

df = df.sort_values("DR", ascending=False).reset_index(drop=True)
df["dr_rank"] = df.index + 1
print(len(df), "players evaluated")

corr, pval = spearmanr(df["dr_rank"], df["wta_rank"])

print(df[["player", "wta_rank", "DR", "dr_rank", "points", "matches"]])
print()
print(f"Spearman correlation between DR rank and WTA rank: {corr:.3f}")
print(f"p-value: {pval:.6f}")

df.to_csv(OUTPUT_DIR / "dr-wta-rank-correlation.csv", index=False)
print(f"\nOutput written to: {OUTPUT_DIR}/dr-wta-rank-corr.csv")

plt.figure(figsize=(8, 6))
plt.scatter(df["DR"], df["wta_rank"])

# Trend line
slope, intercept = np.polyfit(df["DR"], df["wta_rank"], 1)
xLine = np.linspace(df["DR"].min(), df["DR"].max(), 100)
yLine = slope * xLine + intercept

plt.plot(
    xLine, yLine, linestyle="--",
    linewidth=1.5, color ="red")

# WTA rank #1 should appear near the top;
# set 0 exactly at the top edge of the y-axis.
maxWtaRank = df["wta_rank"].max()
plt.ylim(maxWtaRank + 5, 0)

plt.xlabel("Dominance Ratio (DR)")
plt.ylabel("WTA Rank (1 = best)")
plt.title(
    f"Dominance Ratio vs WTA Rank\n"
    f"WTA Top 100, matches >= {MIN_MATCHES}, n={len(df)}" )
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


