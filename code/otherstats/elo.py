# Calculate standard, fixed-K overall Elo ratings

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import pandas as pd

DEFAULT_START_YEAR = 1968
DEFAULT_CUTOFF_DATE = pd.Timestamp("2026-05-25")
DEFAULT_INITIAL_RATING = 1500.0
DEFAULT_K_FACTOR = 20.0

CODE_DIR = Path(__file__).parents[1]
DEFAULT_ARCHIVE_DIR = CODE_DIR / "data" / "tennis-sackmann-archive"
DEFAULT_OUTPUT_DIR = CODE_DIR / "output"

# Qualifying and main-draw files can use separate match_num ranges for the same
# tournament, so round order must precede match_num when the files are combined.
ROUND_ORDER = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
    "QR": 5,
    "RR": 10,
    "R128": 20,
    "R64": 30,
    "R32": 40,
    "R16": 50,
    "QF": 60,
    "SF": 70,
    "F": 80,
    "BR": 90,
}

MATCH_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "tourney_level",
    "tourney_date",
    "match_num",
    "surface",
    "round",
    "winner_name",
    "loser_name",
    "score",
]


# Return player A's expected match score against player B.
#
def expectedScore(ratingA: float, ratingB: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((ratingB - ratingA) / 400.0))

# Return both players' ratings after one match.
#
def updatedRatings(
    ratingA: float,
    ratingB: float,
    scoreA: float,
    kFactor: float,
) -> tuple[float, float]:
    ratingChange = kFactor * (
        scoreA - expectedScore(ratingA, ratingB)
    )
    return ratingA + ratingChange, ratingB - ratingChange

# Load, combine, clean, and chronologically order local WTA matches.
# "cutoffDate" is exclusive.
# The main yearly files contain tour-level main-draw matches.  Set
# "includeQualItf=True" to add the separate qualifying/ITF files.
#
def loadWtaMatches(
    startYear: int = DEFAULT_START_YEAR,
    cutoffDate: str | pd.Timestamp = DEFAULT_CUTOFF_DATE,
    includeQualItf: bool = False,
    archiveDir: str | Path = DEFAULT_ARCHIVE_DIR,
) -> pd.DataFrame:
 
    cutoffTimestamp = pd.Timestamp(cutoffDate).normalize()
    endYear = cutoffTimestamp.year

    if startYear < 1968:
        raise ValueError("The archived WTA match files begin in 1968")
    if startYear > endYear:
        raise ValueError("startYear cannot be later than cutoffDate")

    archivePath = Path(archiveDir)
    wtaPath = archivePath / "wta"
    if not wtaPath.is_dir():
        raise FileNotFoundError(
            f"WTA archive directory not found: {wtaPath}"
        )

    frames = []

    for year in range(startYear, endYear + 1):
        fileNames = [f"wta_matches_{year}.csv"]
        if includeQualItf:
            fileNames.append(f"wta_matches_qual_itf_{year}.csv")

        for fileName in fileNames:
            localPath = wtaPath / fileName
            if not localPath.is_file():
                raise FileNotFoundError(
                    f"WTA archive file not found: {localPath}"
                )

            frame = pd.read_csv(
                localPath,
                usecols=lambda column: column in MATCH_COLUMNS,
                low_memory=False,
            )
            frame["source_file"] = fileName
            frames.append(frame)

    if not frames:
        raise ValueError("No WTA match files were loaded")

    matches = pd.concat(frames, ignore_index=True)
    requiredColumns = {
        "tourney_date",
        "winner_name",
        "loser_name",
    }
    missingColumns = requiredColumns.difference(matches.columns)
    if missingColumns:
        raise ValueError(
            "WTA archive data is missing required columns: "
            f"{sorted(missingColumns)}"
        )

    matches["tourney_date"] = pd.to_datetime(
        matches["tourney_date"].astype("Int64").astype(str),
        format="%Y%m%d",
        errors="raise",
    )

    matches = matches.loc[
        matches["tourney_date"] < cutoffTimestamp
    ].copy()

    matches["winner_name"] = (
        matches["winner_name"].astype("string").str.strip()
    )
    matches["loser_name"] = (
        matches["loser_name"].astype("string").str.strip()
    )

    matches = matches.dropna(
        subset=["winner_name", "loser_name"]
    )
    matches = matches.loc[
        matches["winner_name"].ne("")
        & matches["loser_name"].ne("")
        & matches["winner_name"].ne(matches["loser_name"])
    ].copy()

    # A walkover is not a played match and must not update either rating.
    if "score" in matches.columns:
        score = matches["score"].astype("string").fillna("")
        walkover = score.str.upper().str.contains(
            r"W/O|WALKOVER",
            regex=True,
        )
        matches = matches.loc[~walkover].copy()

    if "tourney_id" not in matches.columns:
        matches["tourney_id"] = ""
    if "match_num" not in matches.columns:
        matches["match_num"] = pd.NA
    if "round" not in matches.columns:
        matches["round"] = pd.NA

    matches["_original_order"] = range(len(matches))
    matches["_match_num"] = pd.to_numeric(
        matches["match_num"],
        errors="coerce",
    )
    matches["_round_order"] = (
        matches["round"].map(ROUND_ORDER).fillna(999)
    )

    # Qualifying and main-draw files can have separate match_num ranges.
    # Therefore, order rounds first and use match_num within each round.
    matches = matches.sort_values(
        [
            "tourney_date",
            "tourney_id",
            "_round_order",
            "_match_num",
            "_original_order",
        ],
        kind="stable",
        na_position="last",
    )

    duplicateColumns = [
        column
        for column in [
            "tourney_id",
            "tourney_date",
            "match_num",
            "winner_name",
            "loser_name",
        ]
        if column in matches.columns
    ]
    matches = matches.drop_duplicates(
        subset=duplicateColumns,
        keep="first",
    )

    return matches.drop(
        columns=["_original_order", "_match_num", "_round_order"]
    ).reset_index(drop=True)


class EloCalc:
    def __init__(
        self,
        matches: pd.DataFrame,
        initialRating: float = DEFAULT_INITIAL_RATING,
        kFactor: float = DEFAULT_K_FACTOR,
    ) -> None:
        if initialRating <= 0:
            raise ValueError("initialRating must be positive")
        if kFactor <= 0:
            raise ValueError("kFactor must be positive")

        requiredColumns = {
            "tourney_date",
            "winner_name",
            "loser_name",
        }
        missingColumns = requiredColumns.difference(matches.columns)
        if missingColumns:
            raise ValueError(
                "matches is missing required columns: "
                f"{sorted(missingColumns)}"
            )

        self.matches = matches.copy()
        self.initialRating = float(initialRating)
        self.kFactor = float(kFactor)
        self._ratingsDf: pd.DataFrame | None = None
        self._historyDf: pd.DataFrame | None = None

    # Construct a calculator from the local WTA archive clone.
    #
    @classmethod
    def fromArchive(
        cls,
        startYear: int = DEFAULT_START_YEAR,
        cutoffDate: str | pd.Timestamp = DEFAULT_CUTOFF_DATE,
        includeQualItf: bool = False,
        archiveDir: str | Path = DEFAULT_ARCHIVE_DIR,
        initialRating: float = DEFAULT_INITIAL_RATING,
        kFactor: float = DEFAULT_K_FACTOR,
    ) -> "EloCalc":
        matches = loadWtaMatches(
            startYear=startYear,
            cutoffDate=cutoffDate,
            includeQualItf=includeQualItf,
            archiveDir=archiveDir,
        )
        return cls(
            matches,
            initialRating=initialRating,
            kFactor=kFactor,
        )

    # Process every match and return current ratings for all players.
    #
    def compute(self) -> pd.DataFrame:
        ratings = defaultdict(lambda: self.initialRating)
        matchCounts = defaultdict(int)
        wins = defaultdict(int)
        losses = defaultdict(int)
        firstMatchDates: dict[str, pd.Timestamp] = {}
        lastMatchDates: dict[str, pd.Timestamp] = {}
        historyRows = []

        for match in self.matches.itertuples(index=False):
            winner = match.winner_name
            loser = match.loser_name
            matchDate = pd.Timestamp(match.tourney_date)

            winnerPreElo = ratings[winner]
            loserPreElo = ratings[loser]
            winnerExpected = expectedScore(
                winnerPreElo,
                loserPreElo,
            )
            winnerPostElo, loserPostElo = updatedRatings(
                winnerPreElo,
                loserPreElo,
                scoreA=1.0,
                kFactor=self.kFactor,
            )

            ratings[winner] = winnerPostElo
            ratings[loser] = loserPostElo
            matchCounts[winner] += 1
            matchCounts[loser] += 1
            wins[winner] += 1
            losses[loser] += 1

            firstMatchDates.setdefault(winner, matchDate)
            firstMatchDates.setdefault(loser, matchDate)
            lastMatchDates[winner] = matchDate
            lastMatchDates[loser] = matchDate

            historyRows.append(
                {
                    "tourney_date": matchDate,
                    "winner_name": winner,
                    "loser_name": loser,
                    "winner_pre_Elo": winnerPreElo,
                    "loser_pre_Elo": loserPreElo,
                    "winner_expected": winnerExpected,
                    "winner_post_Elo": winnerPostElo,
                    "loser_post_Elo": loserPostElo,
                }
            )

        ratingsRows = [
            {
                "player": player,
                "Elo": rating,
                "Elo_matches": matchCounts[player],
                "Elo_wins": wins[player],
                "Elo_losses": losses[player],
                "FIRST_ELO_MATCH_DATE": firstMatchDates[player],
                "LAST_ELO_MATCH_DATE": lastMatchDates[player],
            }
            for player, rating in ratings.items()
        ]

        ratingsDf = pd.DataFrame(ratingsRows)
        if ratingsDf.empty:
            raise ValueError("No played matches were available for Elo")

        ratingsDf["Elo_rank"] = ratingsDf["Elo"].rank(
            ascending=False,
            method="min",
        ).astype("Int64")
        ratingsDf = ratingsDf.sort_values(
            ["Elo_rank", "player"],
            kind="stable",
        ).reset_index(drop=True)

        self._ratingsDf = ratingsDf
        self._historyDf = pd.DataFrame(historyRows)
        return ratingsDf.copy()

    # Return all ratings, computing them first when necessary.
    #
    def ratings(self) -> pd.DataFrame:
        if self._ratingsDf is None:
            return self.compute()
        return self._ratingsDf.copy()

    # Return the auditable match-by-match rating history.
    #
    def history(self) -> pd.DataFrame:
        if self._historyDf is None:
            self.compute()
        return self._historyDf.copy()

    # Return the current Elo summary for one player.
    #
    def elo(
        self,
        playerName: str,
    ) -> tuple[float | None, dict | None, pd.DataFrame | None]:
        playerDf = self.ratings().loc[
            lambda df: df["player"].eq(playerName)
        ].copy()
        if playerDf.empty:
            return None, None, None

        summaryDict = playerDf.iloc[0].to_dict()
        return float(summaryDict["Elo"]), summaryDict, playerDf

    # Return Elo values for a requested player set.
    # Elo_rank is recomputed among the requested players so that it can be
    # merged directly with a common comparison cohort.
    #
    def playersElo(
        self,
        players: list[str],
    ) -> tuple[dict[str, float], pd.DataFrame]:

        requestedDf = pd.DataFrame(
            {"player": players, "_player_order": range(len(players))}
        )
        outputDf = requestedDf.merge(
            self.ratings().drop(columns="Elo_rank"),
            on="player",
            how="left",
        )
        outputDf["Elo_rank"] = outputDf["Elo"].rank(
            ascending=False,
            method="min",
            na_option="keep",
        ).astype("Int64")
        outputDf = outputDf.sort_values(
            "_player_order",
            kind="stable",
        ).drop(columns="_player_order").reset_index(drop=True)

        outputDict = dict(
            zip(
                outputDf.loc[outputDf["Elo"].notna(), "player"],
                outputDf.loc[outputDf["Elo"].notna(), "Elo"],
            )
        )
        return outputDict, outputDf


def _parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute standard Elo ratings from archived WTA results"
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument(
        "--cutoff-date",
        default=f"{DEFAULT_CUTOFF_DATE:%Y-%m-%d}",
        help="Exclusive cutoff date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--initial-rating",
        type=float,
        default=DEFAULT_INITIAL_RATING,
    )
    parser.add_argument(
        "--k-factor",
        type=float,
        default=DEFAULT_K_FACTOR,
    )
    parser.add_argument(
        "--include-qual-itf",
        action="store_true",
        help="Include the separate qualifying and ITF result files",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help="Local tennis-sackmann-archive clone",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parseArguments()
    cutoffDate = pd.Timestamp(args.cutoff_date)

    calc = EloCalc.fromArchive(
        startYear=args.start_year,
        cutoffDate=cutoffDate,
        includeQualItf=args.include_qual_itf,
        archiveDir=args.archive_dir,
        initialRating=args.initial_rating,
        kFactor=args.k_factor,
    )
    ratingsDf = calc.ratings()

    outputPath = args.output
    if outputPath is None:
        lastIncludedDate = cutoffDate - pd.Timedelta(days=1)
        outputPath = (
            DEFAULT_OUTPUT_DIR
            / f"wta-elo-{lastIncludedDate:%Y%m%d}.csv"
        )

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    ratingsDf.to_csv(outputPath, index=False)

    print(
        f"Processed {len(calc.matches):,} matches for "
        f"{len(ratingsDf):,} players"
    )
    print(
        f"Initial rating: {calc.initialRating:g}; "
        f"K-factor: {calc.kFactor:g}"
    )
    print(f"Saved to: {outputPath}")


if __name__ == "__main__":
    main()
