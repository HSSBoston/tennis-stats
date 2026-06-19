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
    edge, summary = computeEdge(name, pointsDeltaV, matches, wDict)
    if edge is None:
        print(f"  {name}: not enough data — skipped.")
        continue
    rows.append(edge)
    print(f"{name:<22}  {edge}")



# edge, summary = computeEdge("Aryna Sabalenka", pointsDeltaV, matches, wDict)
# print(edge)
# pprint(summary)



