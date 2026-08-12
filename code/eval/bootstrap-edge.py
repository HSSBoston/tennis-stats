import pathlib
import sys
PRJ_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PRJ_DIR))

import numpy as np, pandas as pd
from constants import OUTPUT_DIR, RNG_SEED
from dataloader import MCPDataLoader
from edge import EdgeCalc

# Exploratory analysis               500–1,000
# Standard errors                    1,000–2,000
# 95% confidence intervals           2,000–5,000
# Publication-quality tail estimates 5,000–10,000+
# Very small p-values                10,000+
NUM_BOOTSTRAP_SAMPLES = 10
CONFIDENCE_LEVEL = 0.95
MIN_MATCHES = 10

TOP_10 = 10
TOP_20 = 20

# WTA top 100 players as of 06/15/2026
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
    "Maja Chwalinska",
    "Leylah Fernandez",
    "Clara Tauson",
    "Elise Mertens",
    "Emma Navarro",
    "Anastasia Potapova",
    "Marie Bouzkova",
    "Madison Keys",
    "Ann Li",
    "Hailey Baptiste",
    "Emma Raducanu",
    "Xinyu Wang",
    "Donna Vekic",
    "Katerina Siniakova",
    "Alexandra Eala",
    "Cristina Bucsa",
    "Liudmila Samsonova",
    "Jelena Ostapenko",
    "Barbora Krejcikova",
    "Jaqueline Cristian",
    "Maria Sakkari",
    "Laura Siegemund",
    "Janice Tjen",
    "McCartney Kessler",
    "Magdalena Frech",
    "Elisabetta Cocciaretto",
    "Sara Bejlek",
    "Magda Linette",
    "Marketa Vondrousova",
    "Dayana Yastremska",
    "Oleksandra Oliynykova",
    "Petra Marcinko",
    "Maya Joint",
    "Caty McNally",
    "Jessica Bouzas Maneiro",
    "Katie Boulter",
    "Antonia Ruzic",
    "Solana Sierra",
    "Yuliia Starodubtseva",
    "Diane Parry",
    "Zeynep Sonmez",
    "Nikola Bartunkova",
    "Tereza Valentova",
    "Peyton Stearns",
    "Kamilla Rakhimova",
    "Talia Gibson",
    "Shuai Zhang",
    "Panna Udvardy",
    "Daria Kasatkina",
    "Camila Osorio",
    "Anhelina Kalinina",
    "Varvara Gracheva",
    "Kimberly Birrell",
    "Anna Bondar",
    "Daria Snigur",
    "Viktorija Golubic",
    "Renata Zarazua",
    "Tamara Korpatsch",
    "Alycia Parks",
    "Eva Lys",
    "Taylor Townsend",
    "Elsa Jacquemot",
    "Sonay Kartal",
    "Lilli Tagger",
    "Yulia Putintseva",
    "Veronika Erjavec",
    "Karolina Pliskova",
    "Simona Waltert",
    "Oksana Selekhmeteva",
    "Anastasia Zakharova",
    "Maria Timofeeva",
    "Sinja Kraus",
    "Lanlana Tararudee",
    "Ella Seidel",
    "Ashlyn Krueger",
    "Ajla Tomljanovic",
    "Alina Korneeva",
    "Hanne Vandewinkel",
    "Francesca Jones",
    "Emiliana Arango",
]

# Count matches from the original, non-bootstrapped dataset
# Restrict the metadata (in "matches") to match IDs that actually appear in
# the point-by-point data (in "points").
#   points:  original point-by-point data c.f. MCPDataLoader.points
#   matches: original match data (metadata) c.f. MCPDataLoader.matches
# Returns:
#   playerToMatchCounts: Series that maps each player in "players" to her
#     number of matches
#   matchesToBeConsideredDf: Match data (metadata) to be used in the bootstrap analysis 
#
def getOriginalMatchCounts(
    points:  pd.DataFrame,
    matches: pd.DataFrame
)-> tuple[pd.Series, pd.DataFrame]:
    matchIdsInPointData = set(points["match_id"].dropna().unique())

    matchesToBeConsideredDf = matches.loc[
        matches["match_id"].isin(matchIdsInPointData)
    ].copy()

    # Concat two Series; stack the second below the first
    playerNames = pd.concat([matchesToBeConsideredDf["Player 1"],
                             matchesToBeConsideredDf["Player 2"]],
                            ignore_index=True)
    # Create a DataFrame that shows how many times each distinct player name
    # occurs in playerNames; e.g.:
    #   Iga Swiatek     3
    #   Coco Gauff      2
    #   Aryna Sabalenka 1
    counts = playerNames.value_counts()

    playerToMatchCounts = pd.Series(
        {player: int(counts.get(player, 0)) for player in players},
        dtype="int64",
        name="original_matches",
    )
    playerToMatchCounts.index.name = "player"

    return playerToMatchCounts, matchesToBeConsideredDf


def rankEdgeValues(edgeValues):
    """
    Rank players from highest EDGE to lowest EDGE.

    method="min" gives tied players the same best applicable rank.
    Exact ties should be uncommon for EDGE values.
    """
    return edgeValues.rank(
        ascending=False,
        method="min",
        na_option="keep",
    )


def summarizePlayerBootstrap(
    originalEdge,
    originalRanks,
    edgeBootstrapDf,
    rankBootstrapDf,
    matchCounts,
):
    alpha = 1.0 - CONFIDENCE_LEVEL
    lowerQuantile = alpha / 2.0
    upperQuantile = 1.0 - alpha / 2.0

    validEdgeReplicates = edgeBootstrapDf.notna().sum(axis=0)
    validRankReplicates = rankBootstrapDf.notna().sum(axis=0)

    top10Count = rankBootstrapDf.le(TOP_10).sum(axis=0)
    top20Count = rankBootstrapDf.le(TOP_20).sum(axis=0)

    summaryDf = pd.DataFrame(index=originalEdge.index)
    summaryDf.index.name = "player"

    summaryDf["wta_rank_2026_06_15"] = [
        players.index(player) + 1
        for player in summaryDf.index
    ]
    summaryDf["original_matches"] = matchCounts
    summaryDf["original_edge"] = originalEdge
    summaryDf["original_edge_rank"] = originalRanks

    summaryDf["bootstrap_edge_mean"] = edgeBootstrapDf.mean(axis=0)
    summaryDf["bootstrap_edge_median"] = edgeBootstrapDf.median(axis=0)
    summaryDf["bootstrap_edge_bias"] = (
        summaryDf["bootstrap_edge_mean"]
        - summaryDf["original_edge"]
    )
    summaryDf["bootstrap_edge_se"] = edgeBootstrapDf.std(
        axis=0,
        ddof=1,
    )
    summaryDf["edge_ci_lower"] = edgeBootstrapDf.quantile(
        lowerQuantile,
        axis=0,
    )
    summaryDf["edge_ci_upper"] = edgeBootstrapDf.quantile(
        upperQuantile,
        axis=0,
    )
    summaryDf["edge_ci_width"] = (
        summaryDf["edge_ci_upper"]
        - summaryDf["edge_ci_lower"]
    )

    summaryDf["bootstrap_median_rank"] = rankBootstrapDf.median(
        axis=0
    )
    summaryDf["rank_ci_lower"] = rankBootstrapDf.quantile(
        lowerQuantile,
        axis=0,
    )
    summaryDf["rank_ci_upper"] = rankBootstrapDf.quantile(
        upperQuantile,
        axis=0,
    )
    summaryDf["rank_ci_width"] = (
        summaryDf["rank_ci_upper"]
        - summaryDf["rank_ci_lower"]
    )

    # Conditional probabilities: among replicates in which the player
    # received a valid EDGE value and rank.
    summaryDf["probability_top_10_given_valid"] = (
        top10Count
        / validRankReplicates.replace(0, np.nan)
    )
    summaryDf["probability_top_20_given_valid"] = (
        top20Count
        / validRankReplicates.replace(0, np.nan)
    )

    # Unconditional probabilities: missing bootstrap estimates count
    # as failures to finish in the top 10 or top 20.
    summaryDf["probability_top_10_unconditional"] = (
        top10Count / NUM_BOOTSTRAP_SAMPLES
    )
    summaryDf["probability_top_20_unconditional"] = (
        top20Count / NUM_BOOTSTRAP_SAMPLES
    )

    summaryDf["valid_edge_replicates"] = validEdgeReplicates
    summaryDf["valid_rank_replicates"] = validRankReplicates
    summaryDf["availability_rate"] = (
        validEdgeReplicates / NUM_BOOTSTRAP_SAMPLES
    )

    return summaryDf.reset_index()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dl  = MCPDataLoader("w")
    rng = np.random.default_rng(RNG_SEED)

    # Set player eligibility based on the original, non-bootstrapped dataset
    matchCounts, originalMatches = getOriginalMatchCounts(dl.points, dl.matches)

    eligibilityDf = pd.DataFrame(
        {
            "player": players,
            "wta_rank_2026_06_15": range(1, len(players) + 1),
            "original_matches": [
                matchCounts[player] for player in players
            ],
        }
    )
    eligibilityDf["eligible"] = (
        eligibilityDf["original_matches"] >= MIN_MATCHES
    )

    eligiblePlayers = eligibilityDf.loc[
        eligibilityDf["eligible"],
        "player",
    ].tolist()

    if not eligiblePlayers:
        raise ValueError(
            "No players satisfy the eligibility requirement."
        )

    print(
        f"{len(eligiblePlayers)} of {len(players)} players have "
        f"at least {MIN_MATCHES} charted matches."
    )

    # Compute original EDGE values and ranks.
    originalCalc = EdgeCalc(
        dl.points,
        originalMatches,
        saveOutputs=True,
    )
    originalEdgeDict, _ = originalCalc.playersEdge(
        eligiblePlayers
    )

    missingOriginalPlayers = [
        player
        for player in eligiblePlayers
        if player not in originalEdgeDict
    ]

    if missingOriginalPlayers:
        raise ValueError(
            "EDGE could not be calculated from the original data for: "
            + ", ".join(missingOriginalPlayers)
        )

    originalEdge = pd.Series(
        originalEdgeDict,
        dtype="float64",
    ).reindex(eligiblePlayers)
    originalEdge.name = "original_edge"

    originalRanks = rankEdgeValues(originalEdge)
    originalRanks.name = "original_edge_rank"

    originalTop10 = set(
        originalRanks[originalRanks <= TOP_10].index
    )
    originalTop20 = set(
        originalRanks[originalRanks <= TOP_20].index
    )

    edgeReplicates = []
    rankReplicates = []
    globalStabilityRows = []

    for iteration in range(NUM_BOOTSTRAP_SAMPLES):
        bootstrappedPoints, bootstrappedMatches = (
            dl.bootstrap(rng)
        )

        # Re-estimate GWE, event weights, and player EDGE values from
        # this complete match-level bootstrap sample.
        calc = EdgeCalc(
            bootstrappedPoints,
            bootstrappedMatches,
            saveOutputs=False,
        )
        playerEdgeDict, _ = calc.playersEdge(
            eligiblePlayers
        )

        edgeValues = pd.Series(
            playerEdgeDict,
            dtype="float64",
        ).reindex(eligiblePlayers)

        bootstrapRanks = rankEdgeValues(edgeValues)

        edgeReplicates.append(edgeValues)
        rankReplicates.append(bootstrapRanks)

        validRankMask = (
            originalRanks.notna()
            & bootstrapRanks.notna()
        )

        if validRankMask.sum() >= 2:
            # Because these are already rank values, their ordinary
            # correlation is the rank correlation.
            rankCorrelation = originalRanks[
                validRankMask
            ].corr(
                bootstrapRanks[validRankMask]
            )
        else:
            rankCorrelation = np.nan

        bootstrapTop10 = set(
            bootstrapRanks[
                bootstrapRanks <= TOP_10
            ].index
        )
        bootstrapTop20 = set(
            bootstrapRanks[
                bootstrapRanks <= TOP_20
            ].index
        )

        globalStabilityRows.append(
            {
                "bootstrap_iteration": iteration + 1,
                "valid_players": int(validRankMask.sum()),
                "rank_correlation_with_original": (
                    rankCorrelation
                ),
                "top_10_overlap_count": len(
                    originalTop10 & bootstrapTop10
                ),
                "top_10_overlap_rate": (
                    len(originalTop10 & bootstrapTop10)
                    / len(originalTop10)
                    if originalTop10
                    else np.nan
                ),
                "top_20_overlap_count": len(
                    originalTop20 & bootstrapTop20
                ),
                "top_20_overlap_rate": (
                    len(originalTop20 & bootstrapTop20)
                    / len(originalTop20)
                    if originalTop20
                    else np.nan
                ),
            }
        )

        if (
            (iteration + 1) % 100 == 0
            or iteration == 0
            or iteration + 1 == NUM_BOOTSTRAP_SAMPLES
        ):
            print(
                f"Completed {iteration + 1} of "
                f"{NUM_BOOTSTRAP_SAMPLES} bootstrap samples."
            )

    edgeBootstrapDf = pd.DataFrame(edgeReplicates)
    edgeBootstrapDf.index = range(
        1,
        NUM_BOOTSTRAP_SAMPLES + 1,
    )
    edgeBootstrapDf.index.name = "bootstrap_iteration"

    rankBootstrapDf = pd.DataFrame(rankReplicates)
    rankBootstrapDf.index = range(
        1,
        NUM_BOOTSTRAP_SAMPLES + 1,
    )
    rankBootstrapDf.index.name = "bootstrap_iteration"

    globalStabilityDf = pd.DataFrame(globalStabilityRows)

    playerSummaryDf = summarizePlayerBootstrap(
        originalEdge=originalEdge,
        originalRanks=originalRanks,
        edgeBootstrapDf=edgeBootstrapDf,
        rankBootstrapDf=rankBootstrapDf,
        matchCounts=matchCounts.reindex(eligiblePlayers),
    )

    eligibilityDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-eligibility.csv",
        index=False,
    )
    edgeBootstrapDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-value-replicates.csv",
    )
    rankBootstrapDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-rank-replicates.csv",
    )
    playerSummaryDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-player-summary.csv",
        index=False,
    )
    globalStabilityDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-global-stability.csv",
        index=False,
    )

    print("\nBootstrap analysis complete.")
    print(
        playerSummaryDf.sort_values(
            "original_edge_rank"
        ).to_string(index=False)
    )

    print("\nMean global ranking stability:")
    print(
        globalStabilityDf[
            [
                "rank_correlation_with_original",
                "top_10_overlap_rate",
                "top_20_overlap_rate",
            ]
        ].mean()
    )
