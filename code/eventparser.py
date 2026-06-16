import pandas as pd

# Shot letters
#   f = forehand groundstroke (excluding slices, chips, etc.)
#   b = backhand groundstroke (excluding slices, chips, etc.)
#   r = forehand slice (including defensive chips, but not drop shots)
#   s = backhand slice (including defensive chips, but not drop shots)
#   v = forehand volley
#   z = backhand volley
#   o = standard overhead/smash
#   p = "backhand" overhead/smash
#   u = forehand drop shot
#   y = backhand drop shot
#   l = forehand lob
#   m = backhand lob
#   h = forehand half-volley
#   i = backhand half-volley

#   j = forehand swinging volley
#   k = backhand swinging volley
#   t = all trick shots, including behind-the-back, between-the-legs, and "tweeners."
#
#   NOT SUPPORTED: q = any unknown shot

SHOT_LETTERS = set("fbrsvzopuylmhijkt")

# Serve-fault letters
#   n = net (anything that goes into the net, including net cords that are not lets)
#   w = wide (in either direction)
#   d = deep
#   x = both wide and deep
#   e = unknown type of fault
#
#   NOT SUPPORTED: g = foot faults

SERVE_FAULT_CHARS = set("nwdxe")

# *: ace, winner
# @: unforced error
# #: forced error
ENDING_LETTERS = set("*@#")

EVENT_TYPES = [
    "ace_or_winner",   # ace or service winner — server perspective
    "double_fault",    # both serves faulted — server perspective
    "return_error",    # error on the return shot only — returner perspective
    "winner",          # rally winner (incl. return winner) — hitter perspective
    "forced_error",    # forced error drawn — drawer (winner) perspective
    "unforced_error",  # unforced error during rally — errorer perspective
]

# Strip the "c" letters that indicate let serves. 
# 
def stripLetServes(s: str) -> str:
    return s.lstrip("c")

# Returns True if the serve is a fault, False otherwise
#
def isServeFault(code: str) -> bool:
    code = stripLetServes(code)
    if code == "":
        return False
    if any(c in SHOT_LETTERS for c in code):
        return False
    return code[-1] in SERVE_FAULT_CHARS


def classifyEvent(first: str, second: str):
#     if first is None or (isinstance(first, float) and first != first):
#         first = ""
#     if second is None or (isinstance(second, float) and second != second):
#         second = ""
#     first = str(first).strip()
#     second = str(second).strip()
    if first is None or pd.isna(first):
        first = ""
    else:
        first = str(first).strip()
    if second is None or pd.isna(second):
        second = ""
    else:
        second = str(second).strip()
    
    # Exclude "unusual situation" 
    if first in ("", "S", "R", "P", "Q"):
        return None

    f_clean = stripLetServes(first)
    s_clean = stripLetServes(second)
    
    # If the first serve is fault
    if s_clean:
        if isServeFault(s_clean):
            return ("double_fault", "server")
        rally = s_clean
    # if the first serve is not fault
    else:
        rally = f_clean

    if isServeFault(rally):
        return None

    i = 0
    while i < len(rally) and rally[i].isdigit():
        i += 1
    while i < len(rally) and rally[i] in "+=-":
        i += 1
    after_serve = rally[i:]
    if not after_serve:
        return None

    last = after_serve[-1]
    if last not in ENDING_LETTERS:
        return None

    num_rally_shots = sum(1 for c in after_serve if c in SHOT_LETTERS)
    total_shots = 1 + num_rally_shots                  # serve: rally shots
    last_by_server = (total_shots % 2 == 1)            # odd: server hit it

    if num_rally_shots == 0:
        # Pure serve outcome: ace (`*`) or unreturnable serve winner (`#`)
        if last in ("*", "#"):
            return ("ace_or_winner", "server")
        return None

    if last == "*":
        return ("winner", "server" if last_by_server else "returner")

    if last == "@":
        # Unforced error — credited (negatively) to the player who hit it
        if num_rally_shots == 1:
            return ("return_error", "returner")
        return ("unforced_error", "server" if last_by_server else "returner")

    if last == "#":
        # Forced error — credited (positively) to the *opponent* who drew it
        if num_rally_shots == 1:
            return ("return_error", "returner")
        return ("forced_error", "returner" if last_by_server else "server")

    return None


if __name__ == "__main__":
    FirstSecond = [
        ["4b37y1r3n#", None]
    ]

    for fsPair in FirstSecond:
        print( classifyEvent(*fsPair) )


    


