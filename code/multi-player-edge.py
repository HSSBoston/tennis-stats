from dataloader import MCPDataLoader
from edge import EdgeCalc
import pandas as pd
from constants import OUTPUT_DIR
from pprint import pprint

# WTA top 50 players as of 06/15/2026
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
]

dl = MCPDataLoader("w")
calc = EdgeCalc(dl.points, dl.matches)

outputDict, outputDf = calc.playersEdge(players)
print(f"{len(outputDict)} of {len(players)} players evaluated")
pprint(outputDict)

wtaRanks = pd.DataFrame(list(range(1, len(players)+1)),
                        columns = ["wta_rank"])                        
outputDf = pd.concat([outputDf, wtaRanks], axis=1)
outputDf = outputDf.dropna(subset=["player"])
print(outputDf)

outputDf.to_csv(OUTPUT_DIR / "edge-players.csv", index=False)
print(f"Output written to: {OUTPUT_DIR}/edge-players.csv")
