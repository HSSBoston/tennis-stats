from dataloader import MCPDataLoader
from winprob import computeV
from eventweights import computeDeltaV, computeW
from edge import computeEdge
from constants import OUTPUT_DIR

from pprint import pprint

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
    rows.append(edge)
    print(f"{name:<22} {edge:.4f}")

OUT_DIR.mkdir(exist_ok=True)
vDf.to_csv(OUT_DIR / f"v-values.csv")
wDf.to_csv(OUT_DIR / f"w_events_{tour}.csv")


pd.DataFrame([{**{k: v for k, v in r.items() if k != "counts"},
               **{f"count_{k}": v for k, v in r["counts"].items()}}
              for r in rows]).to_csv(
    OUT_DIR / f"X_players_{tour}.csv", index=False)

_save_plot_V(V_stats, f"V(s) — {label} reference population (2020s MCP)",
             OUT_DIR / f"V_states_{tour}.png")
_save_plot_w(w, f"w(e) — average ΔV per event — {label} (2020s MCP)",
             OUT_DIR / f"w_events_{tour}.png")
_save_plot_X(rows, f"X — tennis wOBA-style score — {label}",
             OUT_DIR / f"X_players_{tour}.png")

print(f"\nWrote tables and plots to: {OUT_DIR}/")



