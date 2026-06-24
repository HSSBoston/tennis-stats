from constants import GAME_STATES
import pandas as pd

# Compute the game win expectancy (i.e. probability that the server eventually wins
# the current game for each game state such as "0-0" and "15-0". 
#   df: Point-by-point dataset from the MCP project. Each row represents one point.
#       Obtain the dataset via DataLoader.points.
#   Returns:
#     vDict: Maps each game state to the server's game win expectancy
#     vDf:   DataFrame with columns=["Pts", "game_win_expectancy", ...] where "Pts" means
#            game state
#     df:    Original (MCP) DataFrame + extra columns "server_won_game" (1 or 0) and
#            "game_win_expectancy"
#
def computeGameWinExpectancy(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    # Divide df (point rows) into groups. All points with the same match ID and
    # game number are treated as one tennis game.
    grouped = df.groupby(["match_id", "Gm#"])
    
    # Identify the server and game winner in each game. Obtain a df like: 
    #   match_id   Gm#   server   game_winner
    #   match_A     1      1        1
    #   match_A     2      2        1
    gameResults = grouped.agg(
        server      = ("Svr", "first"),
        game_winner = ("PtWinner", "last"),
    ).reset_index()
    
    # Determine if the server won each game. Insert a new column "server_won", and
    # put 1 or 0 in there (1 if the server won a game). Obtain a df like:
    #   match_id   Gm#   server   game_winner   server_won_game
    #   match_A     1      1        1            1
    #   match_A     2      2        1            0
    gameResults["server_won_game"] = (
        gameResults["server"] == gameResults["game_winner"]
    ).astype(int)
    
    # Add game results to the original df. For example:
    # Before (original df):
    #   match_id   Gm#   Pt   Pts
    #   match_A      1    1   0-0
    #   match_A      1    2  15-0
    #   match_A      1    3  30-0
    # After:
    #   match_id   Gm#   Pt   Pts   server_won_game
    #   match_A      1    1   0-0        1
    #   match_A      1    2  15-0        1
    #   match_A      1    3  30-0        1
    df = df.merge(
        gameResults[["match_id", "Gm#", "server_won_game"]],
        on=["match_id", "Gm#"], # Match rows from the two DataFrames with both match_id and Gm#
        how="left") # Keep all rows from the left DataFrame, which is "df"
    
    # Group all point rows according to their "Pts" game states and select the
    # "server_won_game" column. 
    grouped = df.groupby("Pts")["server_won_game"]

    # Apply two aggregation functions to every game-state group. Obtain a df like:
    # Pts    count  sum
    # 0-0       3    2
    # 15-0      2    2    
    stats = grouped.agg(["count", "sum"])
    
    stats["game_win_expectancy"] = stats["sum"] / stats["count"]
    vDf = stats.reindex(GAME_STATES)
    
    vDict = stats["game_win_expectancy"].to_dict()
    return vDict, vDf, df

    # Examples:
    # vDict = {'0-0': 0.66, '0-15': 0.49, ...}
    # vDf:
    #   Pts  count  sum      game_win_expectancy
    #   0-0  50865  33760.0  0.663718
    #   0-15 21558  10523.0  0.488125
    # df: 
    #   match_id   Gm#   Pt   Pts   server_won_game
    #   match_A      1    1   0-0        1
    #   match_A      1    2  15-0        1
    #   match_A      1    3  30-0        1

if __name__ == "__main__":
    from dataloader import MCPDataLoader
    
    points = MCPDataLoader("w").points
    gweDict, gweDf, pts = computeGameWinExpectancy(points)
    
    print(gweDict)
    print(gweDf)
    
    gweDfSorted = gweDf.sort_values(["game_win_expectancy"])
    print(gweDfSorted)
    
    print(pts)
    
