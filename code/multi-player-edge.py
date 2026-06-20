from dataloader import MCPDataLoader
from winprob import computeV
from eventweights import computeDeltaV, computeW
from edge import computeEdge
from constants import OUTPUT_DIR
from pprint import pprint
import pandas as pd

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
]

dl = MCPDataLoader("w")
points  = dl.points
matches = dl.matches

vDict, vDf, pointsV = computeV(points)
vDfSorted = vDf.sort_values(["game win prob"])
print(vDfSorted)

pointsDeltaV = computeDeltaV(pointsV, vDict)
wDict, wDf = computeW(pointsDeltaV)
print(wDf)
#     print( wDict )


rows = []
for name in players:
    edge, summary = computeEdge(name, pointsDeltaV, matches, wDict)
    if edge is None:
        print(f"  {name}: not enough data — skipped.")
        continue
    rows.append(summary)
    print(f"{name:<22} {edge:.5f}")

OUTPUT_DIR.mkdir(exist_ok=True)
vDf.to_csv(OUTPUT_DIR / f"v-game-expectancy.csv")
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
outputDf.to_csv(OUTPUT_DIR / "edge-players.csv", index=False)
print(f"Output written to: {OUTPUT_DIR}")
