from pathlib import Path
import pandas as pd, numpy as np
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
        self.pointsByMatch = {}
        self.matchesByMatch = {}

        if tour not in {"w", "m"}:
            raise ValueError("Tour must be 'w' or 'm'")
        self.tour = tour

        if pointsFiles == None:
            pointsFiles = [f"charting-{tour}-points-2020s.csv"]
        if len(pointsFiles) == 0:
            raise ValueError("pointsFiles must contain at least one filename")
        matchesFile = f"charting-{tour}-matches.csv"
        
        self.validatePaths(pointsFiles, matchesFile)
        self.loadPoints()
        self.loadMatches()

        # Divide point rows into one DataFrame per match_id
        self.pointsByMatch = {
            matchId: matchPoints
                for matchId, matchPoints in self.points.groupby("match_id", sort=False)
        }
        # Divide metadata (match) rows into one DataFrame per match_id. Each value should
        # contain only one row.
        self.matchesByMatch = {
            matchId: matchRows
                for matchId, matchRows in self.matches.groupby("match_id", sort=False)
        }
        self.validateMetadata()

    def validatePaths(self, pointsFiles: list[str], matchesFile: str) -> None:
        self.pointsPaths = [MCP_DIR / f for f in pointsFiles]
        self.matchesPath = MCP_DIR / matchesFile

        requiredPaths = self.pointsPaths + [self.matchesPath]
        for path in requiredPaths:
            if not path.is_file():
                raise FileNotFoundError(f"Path does not exist: {path}")
        print(f"Input file paths validated: {[f for f in pointsFiles] + [matchesFile]}")

    def validateMetadata(self) -> None:
        for matchId in self.pointsByMatch:
            if matchId not in self.matchesByMatch:
                raise ValueError(f"No metadata found for match: {matchId}")

            if len(self.matchesByMatch[matchId]) != 1:
                raise ValueError(
                    f"Expected one metadata row for {matchId}, "
                    f"found {len(self.matchesByMatch[matchId])}",
                    self.matchesByMatch[matchId])

    def validateBootstrappingConsistency(self, bootstrappedPoints, bootstrappedMatches) -> bool:
        pointMatchIds    = set( bootstrappedPoints["match_id"].unique() )
        metadataMatchIds = set( bootstrappedMatches["match_id"].unique() )
        assert pointMatchIds == metadataMatchIds
        assert ( not bootstrappedMatches["match_id"].duplicated().any() )
        assert len(bootstrappedMatches) == len(dl.pointsByMatch)

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
    
    def bootstrap(self, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
        matchIds = self.points["match_id"].unique()
        sampledIds = rng.choice(matchIds, size=len(matchIds), replace=True)

        sampledPointFrames = []
        sampledMatchFrames = []

        for copyNumber, sourceMatchId in enumerate(sampledIds):
            matchPoints   = self.pointsByMatch[sourceMatchId].copy()
            matchMetadata = self.matchesByMatch[sourceMatchId].copy()

            # Prevent duplicate selections from being grouped together
            bootstrapMatchId = f"{sourceMatchId}__bootstrap_{copyNumber}"
            matchPoints["match_id"]   = bootstrapMatchId
            matchMetadata["match_id"] = bootstrapMatchId
            
            matchPoints["source_match_id"]   = sourceMatchId
            matchMetadata["source_match_id"] = sourceMatchId

            sampledPointFrames.append(matchPoints)
            sampledMatchFrames.append(matchMetadata)
            
            bootstrappedPoints  = pd.concat(sampledPointFrames, ignore_index=True)
            bootstrappedMatches = pd.concat(sampledMatchFrames, ignore_index=True)
            
            if validateBootstrappingConsistency(bootstrappedPoints, bootstrappedMatches):
                return bootstrappedPoints, bootstrappedMatches


if __name__ == "__main__":
    dataLoader = MCPDataLoader("w")
    print( dataLoader.points.head() )

    from constants import RNG_SEED
    rng = np.random.default_rng(RNG_SEED)

    bootstrappedPoints, bootstrappedMatches = dataLoader.bootstrap(rng)
    print(bootstrappedPoints.head())
    print(bootstrappedMatches.head())
    

    
