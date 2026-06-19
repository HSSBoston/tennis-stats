import pandas as pd

# 4: out wide
# 5: body
# 6: down the T
# 0: unknown
SERVE_DIRECTION = set("4560")

# +: Serve-and-volley attempts
# =: Took place at the net
# -: Took place at the baseline
SERVE_MODIFIER = set("+=-")

# n: net (anything that goes into the net, including net cords that are not lets)
# w: wide (in either direction)
# d: deep
# x: both wide and deep
# e: unknown type of fault
# g: foot faults
SERVE_FAULT_LETTERS = set("nwdxeg")

# f: forehand groundstroke (excluding slices, chips, etc.)
# b: backhand groundstroke (excluding slices, chips, etc.)
# r: forehand slice (including defensive chips, but not drop shots)
# s: backhand slice (including defensive chips, but not drop shots)
# v: forehand volley
# z: backhand volley
# o: standard overhead/smash
# p: "backhand" overhead/smash
# u: forehand drop shot
# y: backhand drop shot
# l: forehand lob
# m: backhand lob
# h: forehand half-volley
# i: backhand half-volley
# j: forehand swinging volley
# k: backhand swinging volley
# t: all trick shots, including behind-the-back, between-the-legs, and "tweeners."
# q: any unknown shot
SHOT_LETTERS = set("fbrsvzopuylmhijktq")

# S: Point to the server   (in 1st serve cell)
# R: Point to the returner (in 1st serve cell)
# P: Point penalty against the server   (in 1st serve cell)
# Q: Point penalty against the returner (in 1st serve cell)
UNUSUAL_SITUATIONS = set("SRPQ")
    
# *: ace, winner
# @: unforced error
# #: forced error
ENDING_LETTERS = set("*@#")

EVENT_TYPES = [
    "ace_or_winner",         # service ace or winner — server perspective
    "double_fault",          #                       — server perspective
    "forced_return_error_drawn", # forced return error drawn - server perspective
    "unforced_return_error", #                               — returner perspective
    "winner",                # rally winner (incl. return winner) — hitter perspective
    "forced_error_drawn",    # forced error drawn          — drawer (winner) perspective
    "unforced_error",        # unforced error during rally — errorer perspective
]

# Strip the "c" letters that indicate let serves. 
# 
def stripLetServes(code: str) -> str:
    return code.lstrip("c")

# Returns True if the serve is a fault, False otherwise
#
def isServeFault(code) -> bool:
    code = stripLetServes(code)
    if code == "":
        return False
    if any(c in SHOT_LETTERS for c in code):
        return False
    return code[-1] in SERVE_FAULT_LETTERS

def normalizeCode(code) -> str:
    if pd.isna(code): # if code == None or numpy.NaN 
        return ""
    return str(code).strip()

def classifyEvent(first: str, second: str):
    first  = normalizeCode(first)
    second = normalizeCode(second)
    
    if first == "" or first in UNUSUAL_SITUATIONS:
        return None

    first  = stripLetServes(first)
    second = stripLetServes(second)
    if first == "":
        return None

    # If the second serve is recorded (if the first serve is a fault)
    if second:
        if isServeFault(second):
            return ("double_fault", "server")
        shotSequence = second
    # If the second serve is not recorded (if the first serve is not a fault)
    else:
        shotSequence = first
    
    # In cace the first serve is a fault but second serve is not recoeded
    if isServeFault(shotSequence):
        return None
    
    # Remove the serve information from the beginning of shotSequence, leaving only the
    # shots after the serve. Verify that the remaining code ends with a recognized
    # point-ending symbol.    
    if shotSequence == "" or shotSequence[0] not in SERVE_DIRECTION:
        return None
    i = 1    
    if len(shotSequence) > 1 and shotSequence[1] in SERVE_MODIFIER:
        i = 2
    afterServe = shotSequence[i:]
    if afterServe == "":
        return None

    last = afterServe[-1]
    if last not in ENDING_LETTERS:
        return None

    rallyShotCount = sum(1 for c in afterServe if c in SHOT_LETTERS)
    totalShotCount= 1 + rallyShotCount            # serve + rally shots
    lastShotByServer = (totalShotCount % 2 == 1)  # odd: server hit the last shot

    # Service ace (*) or unreturnable winner (#)
    if rallyShotCount == 0:
        if last in ("*", "#"):
            return ("ace_or_winner", "server")
        return None
    # Winner
    if last == "*":
        if lastShotByServer:
            return ("winner", "server")
        else:
            return ("winner", "returner")
    # Unforced error — credited (negatively) to the player who hit it
    if last == "@":
        if rallyShotCount == 1:
            return ("unforced_return_error", "returner")
        else:
            if lastShotByServer:
                return ("unforced_error", "server")
            else:
                return ("unforced_error", "returner")
    # Forced error — credited (positively) to the opponent who drew it
    if last == "#":
        if rallyShotCount == 1:
            return ("forced_return_error_drawn", "server")
        else:
            if lastShotByServer:
                return ("forced_error_drawn", "returner")
            else:
                return ("forced_error_drawn", "server")

    return None


if __name__ == "__main__":
    # ace on first serve
    assert classifyEvent("6*", None) == ("ace_or_winner","server")
    # unreturnable winner on first serve
    assert classifyEvent("4#", None) == ("ace_or_winner","server")
    
    # ace on second serve
    assert classifyEvent("4w", "5*") == ("ace_or_winner", "server")
    # unreturnable winner on second serve
    assert classifyEvent("6d", "6#") == ("ace_or_winner", "server")
    
    # double fault
    assert classifyEvent("4w", "6d")  == ("double_fault", "server")
    assert classifyEvent("4n", "5n") == ("double_fault", "server")

    # Forced error on return
    assert classifyEvent("6b2n#", "") == ("forced_return_error_drawn", "server")
    assert classifyEvent("6f#", "")  == ("forced_return_error_drawn", "server")

    # Unforced error on return
    assert classifyEvent("4f3d@", "")  == ("unforced_return_error", "returner")
    assert classifyEvent("6w", "4b2n@") == ("unforced_return_error", "returner")
    
    # rally
    assert classifyEvent("6d", "6b27b3b3b3b3b3x@") == ("unforced_error", "server")
    assert classifyEvent("4b37y1r3n#", None) == ("forced_error_drawn", "server")
    assert classifyEvent("4+f28b3@", None)   == ("unforced_error", "server")
    assert classifyEvent("4n", "4b38y1*")    == ("winner", "server")
    assert classifyEvent("4f37f3*", "")      == ("winner", "server")
    assert classifyEvent("4f37b1f3*", "")    == ("winner", "returner")
    assert classifyEvent("4f37b1f3n@", "")   == ("unforced_error", "returner")
    assert classifyEvent("4f37b1f3n#", "")   == ("forced_error_drawn", "server")

