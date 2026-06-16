from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).parent
MCP_DIR = CURRENT_DIR / "data" / "tennis_MatchChartingProject"
OUTPUT_DIR = CURRENT_DIR / "output"





def run(tour: str = "w",
        pointsFiles: Optional[list] = None,
        players: Optional[list] = None) -> None:

    if pointsFiles == None:
        pointsFiles = [f"charting-{tour}-points-2020s.csv"]
        
    if players == None:
        if tour == "w":
            players = ["Iga Swiatek", "Aryna Sabalenka", "Coco Gauff",
             "Elena Rybakina", "Jessica Pegula", "Mirra Andreeva",
             "Ons Jabeur", "Maria Sakkari", "Karolina Pliskova"]
        elif tour == "m":
            players = ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
             "Daniil Medvedev", "Stefanos Tsitsipas",
             "Alexander Zverev", "Andrey Rublev", "Casper Ruud",
             "Hubert Hurkacz", "Rafael Nadal"]

    pointsPaths = [MCP_DIR / f for f in pointsFiles]
    matchesPath = MCP_DIR / f"charting-{tour}-matches.csv"
    requiredPaths = pointsPaths + [matchesPath]
    
    for path in requiredPaths:
#         if path.is_file():
#             print(path)
        if not path.is_file():
            raise FileNotFoundError(f"Path does not exist: {path}") 


if __name__ == "__main__":
    run("w")
    