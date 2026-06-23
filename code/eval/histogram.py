import pathlib, sys
PRJ_DIR = pathlib.Path(__file__).parents[1] # 2 levels up
sys.path.append(str(PRJ_DIR))

from dataloader import MCPDataLoader
from edge import EdgeCalc
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

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

MIN_MATCHES = 15
EDGE_SCALE = 1000

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
print(f"{playersCountBefore} players evaluated")

scaledEdgeValues = outputDf["EDGE"] * EDGE_SCALE
      
plt.figure(figsize=(8, 5))
plt.hist(edgeValues, bins=12, edgecolor="black")

plt.axvline(scaledEdgeValues.mean(), linestyle="--", linewidth=1.5,
            label=f"Mean = {scaledEdgeValues.mean():.2f}")

plt.axvline(scaledEdgeValues.median(), linestyle=":", linewidth=1.5,
            label=f"Median = {scaledEdgeValues.median():.2f}")

plt.title(f"Distribution of Player EDGE Values\nWTA Top 100, matches >= {MIN_MATCHES}")
plt.xlabel(f"EDGE × {EDGE_SCALE}")
plt.ylabel("Number of players")
plt.legend()

plt.tight_layout()
plt.show()

