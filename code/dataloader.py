from pathlib import Path
import pandas as pd
from constants import GAME_STATES, MCP_DIR

class MCPDataLoader:
    def __init__(self,
        tour: str = "w",
        pointsFiles: list[str] | None = None
    ) -> None:

        self.tour: str = ""
        self.pointsPaths: list[Path] = []
        self.matchesPath: Path
        self.points = pd.DataFrame()
        self.matches = pd.DataFrame()

        if tour not in {"w", "m"}:
            raise ValueError("tour must be 'w' or 'm'")

        if pointsFiles == None:
            pointsFiles = [f"charting-{tour}-points-2020s.csv"]
        if len(pointsFiles) == 0:
            raise ValueError("pointsFiles must contain at least one filename")
        matchesFile = f"charting-{tour}-matches.csv"
        
        self.tour = tour

        self.verifyPaths(pointsFiles, matchesFile)
        self.loadPoints()
        self.loadMatches()

    def verifyPaths(self, pointsFiles: list[str], matchesFile: str) -> None:
        self.pointsPaths = [MCP_DIR / f for f in pointsFiles]
        self.matchesPath = MCP_DIR / matchesFile

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
        df = df.loc[ df["Pts"].isin(GAME_STATES) ]
        df = df.sort_values(["match_id", "Pt"]).reset_index(drop=True)
            # Sort rows by "match_id" and then "Pt"
            # drop=True: Discard old index numbers
        self.points = df
        print(len(self.points), "points loaded")
        
    def loadMatches(self) -> None:
        self.matches = pd.read_csv(self.matchesPath, dtype=str)



if __name__ == "__main__":
    dataLoader = MCPDataLoader("w")
    print(dataLoader.points.head())


