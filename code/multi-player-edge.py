from dataloader import MCPDataLoader
from winprob import computeV
from eventweights import computeDeltaV, computeW
from edge import computeEdge
from pprint import pprint

playersW = ["Iga Swiatek", "Aryna Sabalenka", "Coco Gauff",
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

edge, summary = computeEdge("Aryna Sabalenka", pointsDeltaV, matches, wDict)
print(edge)
pprint(summary)



