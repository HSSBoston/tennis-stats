from constants import GAME_STATES, playersW, playersM
from parser import Parser
import pandas as pd

# Compute the probability that the server eventually wins the current game for each
# game state such as "0-0" and "15-0". 
#   df: Point-by-point dataset. Each row represents one point. C.f. Parser.points.
#   
def computeV(df: pd.DataFrame):
    # Divide df (point rows) into groups. All points with the same match ID and
    # game number are treated as one tennis game.
    grouped = df.groupby(["match_id", "Gm#"])
    
    # Identify the server and game winner in each game. Obtain a df like: 
    #   match_id   Gm#   server   game_winner
    #   match_A      1       1            1
    #   match_A      2       2            1
    gameResults = grouped.agg(
        server = ("Svr", "first"),
        game_winner = ("PtWinner", "last"),
    ).reset_index()
    
    # Determine if the server won each game. Insert a new column "server_won", and
    # put 1 or 0 (1 if the server won a game). Obtain a df like:
    #   match_id   Gm#   server   game_winner   server_won_game
    #   match_A      1       1         1            1
    #   match_A      2       2         1            0
    gameResults["server_won_game"] = (
        gameResults["server"] == gameResults["game_winner"]
    ).astype(int)
    
    # Attach the game result to the original df. For example:
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
        on=["match_id", "Gm#"],
        how="left")
    
    # Group all point rows according to their "Pts" game states and select the
    # "server_won_game" column. 
    grouped = df.groupby("Pts")["server_won_game"]

    # Apply two aggregation functions to every game-state group. Obtain a df like:
    #        count  sum
    # Pts
    # 0-0       3    2
    # 15-0      2    2    
    stats = grouped.agg(["count", "sum"])
    
    stats["game win probability"] = stats["sum"] / stats["count"]
    stats = stats.reindex(GAME_STATES)
    
    Vdict = stats["game win probability"].to_dict()
    return Vdict, df, stats


if __name__ == "__main__":
    points = Parser("w", playersW).points
    Vdict, df, stats = computeV(points)
    print(Vdict)
    print(df)
    print(stats)


