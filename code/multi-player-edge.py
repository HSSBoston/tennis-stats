from dataloader import MCPDataLoader
from winprob import computeV
from eventweights import computeDeltaV, computeW
from edge import computeEdge
from pprint import pprint

players = ["Iga Swiatek", "Aryna Sabalenka", "Coco Gauff",
                 "Elena Rybakina", "Jessica Pegula", "Mirra Andreeva",
                 "Ons Jabeur", "Maria Sakkari", "Karolina Pliskova"]

# playersM = ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
#                  "Daniil Medvedev", "Stefanos Tsitsipas",
#                  "Alexander Zverev", "Andrey Rublev", "Casper Ruud",
#                  "Hubert Hurkacz", "Rafael Nadal"]

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
    res = compute_player_X(name, pts, matches, w_dict)
    if res is None:
        print(f"  {name}: not enough data — skipped.")
        continue
    rows.append(res)
    print(f"  {name:<22}  X = {res['X']:+.4f}   "
          f"(N = {res['N']:,} pts, {res['matches']} matches)")

if not rows:
    print("\nNo players had usable data. Exiting.")
    return



edge, summary = computeEdge("Aryna Sabalenka", pointsDeltaV, matches, wDict)
print(edge)
pprint(summary)



