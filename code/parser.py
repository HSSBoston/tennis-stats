from pathlib import Path
import pandas as pd
from constants import GAME_STATES

class Parser:
    CURRENT_DIR = Path(__file__).parent
    MCP_DIR     = CURRENT_DIR / "data" / "tennis_MatchChartingProject"
    OUTPUT_DIR  = CURRENT_DIR / "output"
        
    def __init__(self,
        tour: str = "w",
        players: list[str] | None = None,
        pointsFiles: list[str] | None = None
    ) -> None:

        self.pointsPaths: list[Path]
        self.matchesPath: Path
        self.points = pd.DataFrame
        self.matches = pd.DataFrame

        if tour not in {"w", "m"}:
            raise ValueError("tour must be 'w' or 'm'")

        if pointsFiles == None:
            pointsFiles = [f"charting-{tour}-points-2020s.csv"]
        if len(pointsFiles) == 0:
            raise ValueError("pointsFiles must contain at least one filename")
        matchesFile = f"charting-{tour}-matches.csv"

        self.verifyPaths(tour, pointsFiles, matchesFile)
        self.loadPoints()
        self.loadMatches()
#         self.verifyPlayers()

    def verifyPaths(self, tour: str, pointsFiles: list[str], matchesFile: str) -> None:
        self.pointsPaths = [self.MCP_DIR / f for f in pointsFiles]
        self.matchesPath = self.MCP_DIR / matchesFile

        requiredPaths = self.pointsPaths + [self.matchesPath]
        for path in requiredPaths:
            if not path.is_file():
                raise FileNotFoundError(f"Path does not exist: {path}")
        print(f"Input file paths verified: {[f for f in pointsFiles] + [matchesFile]}")

    def loadPoints(self) -> None:
        frames = [pd.read_csv(p, dtype=str) for p in self.pointsPaths]
        df = pd.concat(frames, axis=0, ignore_index=True)
        for col in ("Pt", "Svr", "PtWinner"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                # errors="coerce": replacing invalid value with NaN, not raising an exception
                # "Int64": converting NaN to Pandas' nullable integer (pd.NA)
#         print(df.head())
        self.points = df[df["Pts"].isin(GAME_STATES)]
        print(self.points.head())
        
    def loadMatches(self) -> None:
        self.matches = pd.read_csv(self.matchesPath, dtype=str)



if __name__ == "__main__":
    playersW = ["Iga Swiatek", "Aryna Sabalenka", "Coco Gauff",
                     "Elena Rybakina", "Jessica Pegula", "Mirra Andreeva",
                     "Ons Jabeur", "Maria Sakkari", "Karolina Pliskova"]
    playersM = ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner",
                     "Daniil Medvedev", "Stefanos Tsitsipas",
                     "Alexander Zverev", "Andrey Rublev", "Casper Ruud",
                     "Hubert Hurkacz", "Rafael Nadal"]

    parser = Parser("w", playersW) 
