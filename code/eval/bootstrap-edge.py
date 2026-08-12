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

# WTA top 100 players as of 05/25/2026
players = [
    "Aryna Sabalenka",
    "Elena Rybakina",
    "Iga Swiatek",
    "Coco Gauff",
    "Jessica Pegula",
    "Amanda Anisimova",
    "Elina Svitolina",
    "Mirra Andreeva",
    "Victoria Mboko",
    "Karolina Muchova",
    "Belinda Bencic",
    "Linda Noskova",
    "Jasmine Paolini",
    "Ekaterina Alexandrova",
    "Marta Kostyuk",
    "Naomi Osaka",
    "Iva Jovic",
    "Sorana Cirstea",
    "Madison Keys",
    "Clara Tauson",
    "Elise Mertens",
    "Leylah Fernandez",
    "Diana Shnaider",
    "Anna Kalinskaya",
    "Emma Navarro",
    "Hailey Baptiste",
    "Liudmila Samsonova",
    "Marie Bouzkova",
    "Ann Li",
    "Anastasia Potapova",
    "Jelena Ostapenko",
    "Jaqueline Cristian",
    "Cristina Bucsa",
    "Xin Yu Wang", # Xinyu Wang
    "Sara Bejlek",
    "Katerina Siniakova",
    "Alexandra Eala",
    "Elisabetta Cocciaretto",
    "Emma Raducanu",
    "Janice Tjen",
    "Barbora Krejcikova",
    "Tereza Valentova",
    "Lois Boisson",
    "Marketa Vondrousova",
    "Dayana Yastremska",
    "Magdalena Frech",
    "Laura Siegemund",
    "Mccartney Kessler",
    "Maria Sakkari",
    "Jessica Bouzas Maneiro",
    "Petra Marcinko",
    "Maya Joint",
    "Daria Kasatkina",
    "Tatjana Maria",
    "Yuliia Starodubtseva",
    "Qinwen Zheng",
    "Anna Bondar",
    "Talia Gibson",
    "Panna Udvardy",
    "Anhelina Kalinina",
    "Shuai Zhang",
    "Sonay Kartal",
    "Caty Mcnally",
    "Antonia Ruzic",
    "Oleksandra Oliynykova",
    "Zeynep Sonmez",
    "Elsa Jacquemot",
    "Solana Sierra",
    "Nikola Bartunkova",
    "Varvara Gracheva",
    "Katie Boulter",
    "Donna Vekic",
    "Magda Linette",
    "Renata Zarazua",
    "Taylor Townsend",
    "Yulia Putintseva",
    "Elena Gabriela Ruse", # Elena-Gabriela Ruse
    "Peyton Stearns",
    "Alycia Parks",
    "Anastasia Zakharova",
    "Eva Lys",
    "Viktorija Golubic",
    "Kimberly Birrell",
    "Veronika Erjavec",
    "Veronika Kudermetova",
    "Camila Osorio",
    "Sofia Kenin",
    "Oksana Selekhmeteva",
    "Kamilla Rakhimova",
    "Lilli Tagger",
    "Simona Waltert",
    "Diane Parry",
    "Daria Snigur",
    "Emiliana Arango",
    "Tamara Korpatsch",
    "Ella Seidel",
    "Lanlana Tararudee",
    "Sinja Kraus",
    "Hanne Vandewinkel",
    "Ajla Tomljanovic"
]

# Count the number of matches each player in "players" (e.g. WTA top 100 players) played.
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


# Convert player EDGE values to rank values
#   edgeValues: Series that maps players to their EDGE values
# Returns:
#   Series that maps players to their rank values in float (1.0: best).
#
def rankEdgeValues(edgeValues):
    return edgeValues.rank(
        ascending=False, # Following the descending order of EDGE values (highest EDGE: best rank)
        method="min",    # gives tied players the same best applicable rank
        na_option="keep" )


def summarizePlayerBootstrap(
    originalEdge,
    originalRanks,
    edgeBootstrapDf,
    rankBootstrapDf,
    matchCounts
) -> None:
    alpha = 1.0 - CONFIDENCE_LEVEL
    lowerQuantile = alpha / 2.0
    upperQuantile = 1.0 - alpha / 2.0

    validEdgeReplicates = edgeBootstrapDf.notna().sum(axis=0)
    validRankReplicates = rankBootstrapDf.notna().sum(axis=0)

    top10Count = rankBootstrapDf.le(10).sum(axis=0)
    top20Count = rankBootstrapDf.le(20).sum(axis=0)

    summaryDf = pd.DataFrame(index=originalEdge.index)
    summaryDf.index.name = "player"

    summaryDf["wta_rank"]           = [ players.index(player) + 1 for player in summaryDf.index ]
    summaryDf["original_matches"]   = matchCounts
    summaryDf["original_edge"]      = originalEdge
    summaryDf["original_edge_rank"] = originalRanks

    summaryDf["bootstrap_edge_mean"]   = edgeBootstrapDf.mean(axis=0)
    summaryDf["bootstrap_edge_median"] = edgeBootstrapDf.median(axis=0)
    summaryDf["bootstrap_edge_bias"]   = summaryDf["bootstrap_edge_mean"] - summaryDf["original_edge"]
    summaryDf["bootstrap_edge_se"]     = edgeBootstrapDf.std(axis=0, ddof=1)
    summaryDf["edge_ci_lower"]         = edgeBootstrapDf.quantile(lowerQuantile, axis=0)
    summaryDf["edge_ci_upper"]         = edgeBootstrapDf.quantile(upperQuantile, axis=0)
    summaryDf["edge_ci_width"]         = summaryDf["edge_ci_upper"] - summaryDf["edge_ci_lower"]
    summaryDf["bootstrap_rank_median"] = rankBootstrapDf.median(axis=0)
    summaryDf["rank_ci_lower"]         = rankBootstrapDf.quantile(lowerQuantile, axis=0)
    summaryDf["rank_ci_upper"]         = rankBootstrapDf.quantile(upperQuantile, axis=0)
    summaryDf["rank_ci_width"]         = summaryDf["rank_ci_upper"] - summaryDf["rank_ci_lower"]

    # Conditional probabilities: among replicates in which the player
    # received a valid EDGE value and rank.
    summaryDf["probability_top_10_given_valid"] = top10Count / validRankReplicates.replace(0, np.nan)
    summaryDf["probability_top_20_given_valid"] = top20Count / validRankReplicates.replace(0, np.nan)

    # Unconditional probabilities: missing bootstrap estimates count
    # as failures to finish in the top 10 or top 20.
    summaryDf["probability_top_10_unconditional"] = top10Count / NUM_BOOTSTRAP_SAMPLES
    summaryDf["probability_top_20_unconditional"] = top20Count / NUM_BOOTSTRAP_SAMPLES

    summaryDf["valid_edge_replicates"] = validEdgeReplicates
    summaryDf["valid_rank_replicates"] = validRankReplicates
    summaryDf["availability_rate"]     = validEdgeReplicates / NUM_BOOTSTRAP_SAMPLES

    return summaryDf.reset_index()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dl  = MCPDataLoader("w")
    rng = np.random.default_rng(RNG_SEED)

    # Count the number of matches each of WTA top 100 players played based on the
    # original, non-bootstrapped dataset
    playerToMatchCounts, originalMatches = getOriginalMatchCounts(dl.points, dl.matches)

    eligibilityDf = pd.DataFrame({
        "player":           players,
        "wta_rank":         range(1, len(players) + 1),
        "original_matches": [playerToMatchCounts[player] for player in players] })
    
    # Determine which players are eligible for bootstrap analysis (>= MIN_MATCHES)
    eligibilityDf["eligible"] = eligibilityDf["original_matches"] >= MIN_MATCHES
    eligiblePlayersList = eligibilityDf.loc[
        eligibilityDf["eligible"] == True,
        "player"
    ].tolist()

    if not eligiblePlayersList:
        raise ValueError("No players satisfy the eligibility requirement.")
    print(
        f"{len(eligiblePlayersList)} of {len(players)} players have "
        f"at least {MIN_MATCHES} charted matches.")

    # Compute original EDGE values and ranks based on the original, non-bootstrapped dataset
    originalCalc = EdgeCalc( dl.points, originalMatches, saveOutputs=True )
    originalEdgeDict, _ = originalCalc.playersEdge(eligiblePlayersList)

    missingOriginalPlayers = [ player for player in eligiblePlayersList
                                 if player not in originalEdgeDict ]
    if missingOriginalPlayers:
        raise ValueError("EDGE could not be calculated from the original data for: "
                         + ", ".join(missingOriginalPlayers) )

    originalEdge  = pd.Series(originalEdgeDict, dtype="float64").reindex(eligiblePlayersList)
    originalRanks = rankEdgeValues(originalEdge)

    # Select EDGE top 10 and 20 player names
    originalTop10 = set( originalRanks[originalRanks <= 10].index )
    originalTop20 = set( originalRanks[originalRanks <= 20].index )

    edgeReplicates = []
    rankReplicates = []
    globalStabilityRows = []

    for iteration in range(NUM_BOOTSTRAP_SAMPLES):
        bootstrappedPoints, bootstrappedMatches = dl.bootstrap(rng)

        # Re-compute GWE, event weights, and player EDGE values from
        # this bootstrapped point-by-point data.
        calc = EdgeCalc( bootstrappedPoints, bootstrappedMatches, saveOutputs=False )
        playerEdgeDict, _ = calc.playersEdge( eligiblePlayersList )

        edgeValues = pd.Series(playerEdgeDict, dtype="float64").reindex(eligiblePlayersList)
        bootstrapRanks = rankEdgeValues(edgeValues)

        edgeReplicates.append(edgeValues)
        rankReplicates.append(bootstrapRanks)

        # Calculates the correlation between the original EDGE rankings and
        # one bootstrap sample’s EDGE rankings, using only players with valid ranks in both
        validRankMask = originalRanks.notna() & bootstrapRanks.notna()
        if validRankMask.sum() >= 2:
            # sum() returns the number of players with valid ranks in both Series.
            # Because these are already rank values, their ordinary correlation is
            # the rank correlation.
            rankCorrelation = originalRanks[validRankMask].corr(
                bootstrapRanks[validRankMask])
        else:
            rankCorrelation = np.nan

        bootstrapTop10 = set( bootstrapRanks[bootstrapRanks <= 10].index )
        bootstrapTop20 = set( bootstrapRanks[bootstrapRanks <= 20].index )

        globalStabilityRows.append( {
            "bootstrap_iteration":            iteration + 1,
            "valid_players":                  int(validRankMask.sum()),
            "rank_correlation_with_original": rankCorrelation,
            "top_10_overlap_count": len(originalTop10 & bootstrapTop10),
            "top_10_overlap_rate":  (len(originalTop10 & bootstrapTop10) / len(originalTop10)
                                     if originalTop10 else np.nan),
            "top_20_overlap_count": len(originalTop20 & bootstrapTop20),
            "top_20_overlap_rate":  (len(originalTop20 & bootstrapTop20) / len(originalTop20)
                                     if originalTop20 else np.nan)
            } )

        if( (iteration+1)%100==0 or iteration==0 or (iteration+1)==NUM_BOOTSTRAP_SAMPLES):
            print(f"Completed {iteration+1} of {NUM_BOOTSTRAP_SAMPLES} bootstrap samples.")

    edgeBootstrapDf = pd.DataFrame(edgeReplicates)
    edgeBootstrapDf.index = range(1, NUM_BOOTSTRAP_SAMPLES + 1)
    edgeBootstrapDf.index.name = "bootstrap_iteration"

    rankBootstrapDf = pd.DataFrame(rankReplicates)
    rankBootstrapDf.index = range(1, NUM_BOOTSTRAP_SAMPLES + 1)
    rankBootstrapDf.index.name = "bootstrap_iteration"

    globalStabilityDf = pd.DataFrame(globalStabilityRows)

    playerSummaryDf = summarizePlayerBootstrap(
        originalEdge=originalEdge,
        originalRanks=originalRanks,
        edgeBootstrapDf=edgeBootstrapDf,
        rankBootstrapDf=rankBootstrapDf,
        matchCounts=playerToMatchCounts.reindex(eligiblePlayersList) )

    eligibilityDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-eligibility.csv", index=False)
    edgeBootstrapDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-value-replicates.csv")
    rankBootstrapDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-rank-replicates.csv")
    playerSummaryDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-player-summary.csv", index=False)
    globalStabilityDf.to_csv(
        OUTPUT_DIR / "bootstrap-edge-global-stability.csv", index=False)

    print("\nBootstrap analysis complete.")
#    print( playerSummaryDf.sort_values("original_edge_rank").to_string(index=False) )

    print("\nMean global ranking stability:")
    print(globalStabilityDf[ ["rank_correlation_with_original",
                              "top_10_overlap_rate",
                              "top_20_overlap_rate"] ].mean() )
