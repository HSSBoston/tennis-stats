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
]

dl = MCPDataLoader("w")
points  = dl.points
matches = dl.matches

gweDict, gweDf, pointsGwe = computeGameWinExpectancy(points)
gweDfSorted = gweDf.sort_values(["game win expectancy"])
print(gweDfSorted)

pointsDeltaGwe = computeDeltaGameWinExpectancy(pointsGwe, gweDict)
wDict, wDf = computeEventWeights(pointsDeltaGwe)
print(wDf)
#     print( wDict )


rows = []
for name in players:
    edge, summary = computeEdge(name, pointsDeltaGwe, matches, wDict)
    if edge is None:
        print(f"  {name}: not enough data — skipped.")
        continue
    rows.append(summary)
    print(f"{name:<22} {edge:.5f}")

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
        **{key: value
           for key, value in r.items() if (key != "event_counts") or (key != "event_EDGE")},
        **{ev+"edge": evEdge for ev, evEdge in r["event_EDGE"].items()},
        **{ev+"count": count for ev, count in r["event_counts"].items()}
    }
    for r in rows
])
outputDf.to_csv(OUTPUT_DIR / "edge-players.csv", index=False)
print(f"Output written to: {OUTPUT_DIR}/edge-players.csv")
