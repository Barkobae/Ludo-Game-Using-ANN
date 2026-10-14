#!/usr/bin/env python3
"""
Ludo dataset generator for ANN training (imitation learning / value learning).

Simulates 4-player Ludo games. Every decision (a dice roll with >=2 legal tokens)
is logged as one row: a numeric feature vector from the mover's point of view,
a legal-move mask, the expert (heuristic) label, the move actually played,
and whether that player eventually won the game.

Usage:
    python ludo_dataset_generator.py --games 4000 --seed 42 --out ludo_dataset.csv
    python ludo_dataset_generator.py --games 4000 --keep-forced   # also log forced moves

Rules implemented
  * 52-square shared track, 4 players, 4 tokens each, start squares 0/13/26/39
  * Token progress: -1 = yard, 0..50 = main track, 51..55 = home column, 56 = finished
  * A 6 is needed to leave the yard; exact roll needed to finish
  * Safe squares: the 4 start squares + 4 star squares (8, 21, 34, 47) + home column
  * Capture: landing on an opponent on a non-safe track square sends it to the yard
  * Extra turn on a 6 or on a capture; three 6s in a row forfeit the third move
  * First player to get all 4 tokens home wins
"""
import argparse
import csv
import random

START = [0, 13, 26, 39]
SAFE = {0, 13, 26, 39, 8, 21, 34, 47}
TRACK_LAST = 50   # last main-track progress value
FINISH = 56
N = 52
MAX_TURNS = 3000


# ---------------------------------------------------------------- engine
def abs_sq(pl, p):
    """Absolute track square for progress p, or None if not on the shared track."""
    return (START[pl] + p) % N if 0 <= p <= TRACK_LAST else None


def is_legal(p, dice):
    if p == -1:
        return dice == 6
    return p + dice <= FINISH


def new_progress(p, dice):
    return 0 if p == -1 else p + dice


def opponent_tokens_on_track(pos, pl):
    out = []
    for o in range(4):
        if o == pl:
            continue
        for t in range(4):
            a = abs_sq(o, pos[o][t])
            if a is not None:
                out.append((a, o, t))
    return out


def captured_at(pos, pl, sq):
    """Opponent tokens that would be captured by landing on absolute square sq."""
    if sq is None or sq in SAFE:
        return []
    return [(o, t) for (a, o, t) in opponent_tokens_on_track(pos, pl) if a == sq]


def danger_count(pos, pl, sq):
    """Number of opponent tokens 1..6 squares behind sq (0 if sq is safe/off-track)."""
    if sq is None or sq in SAFE:
        return 0
    c = 0
    for (a, _, _) in opponent_tokens_on_track(pos, pl):
        if 1 <= (sq - a) % N <= 6:
            c += 1
    return c


def nearest_dist(pos, pl, sq, direction):
    """Distance (1..12) to the nearest opponent behind (-1) / ahead (+1); else None."""
    best = None
    for (a, _, _) in opponent_tokens_on_track(pos, pl):
        d = (sq - a) % N if direction < 0 else (a - sq) % N
        if 1 <= d <= 12 and (best is None or d < best):
            best = d
    return best


# ---------------------------------------------------------------- features
def token_info(pos, pl, t, dice):
    """Per-token move analysis for the current dice roll."""
    p = pos[pl][t]
    info = dict(legal=0, capture=0, cap_prog=0.0, finish=0, leave=0, enter_home=0,
                safe_after=0, danger_now=0.0, danger_after=0.0, threat=0.0, target=0.0)
    cur = abs_sq(pl, p)
    if cur is not None:
        info["danger_now"] = min(danger_count(pos, pl, cur), 3) / 3
        d = nearest_dist(pos, pl, cur, -1)
        info["threat"] = 0.0 if d is None else 1 - (d - 1) / 12
        d = nearest_dist(pos, pl, cur, +1)
        info["target"] = 0.0 if d is None else 1 - (d - 1) / 12
    if not is_legal(p, dice):
        return info
    np_ = new_progress(p, dice)
    sq = abs_sq(pl, np_)
    info["legal"] = 1
    info["leave"] = int(p == -1)
    info["finish"] = int(np_ == FINISH)
    info["enter_home"] = int(p <= TRACK_LAST and np_ > TRACK_LAST and np_ != FINISH)
    info["safe_after"] = int(sq is None or sq in SAFE)
    caps = captured_at(pos, pl, sq)
    if caps:
        info["capture"] = 1
        info["cap_prog"] = max(pos[o][tt] for o, tt in caps) / FINISH
    info["danger_after"] = min(danger_count(pos, pl, sq), 3) / 3
    return info


def build_features(pos, pl, dice):
    infos = [token_info(pos, pl, t, dice) for t in range(4)]
    f = {}
    for d in range(1, 7):
        f[f"dice_{d}"] = int(dice == d)
    for t in range(4):
        f[f"my_prog_{t}"] = round((pos[pl][t] + 1) / 57, 4)
    for key in ("legal", "capture", "finish", "leave", "enter_home", "safe_after",
                "danger_now", "danger_after", "threat", "target"):
        for t in range(4):
            f[f"{key}_{t}"] = round(infos[t][key], 4)
    # Opponents (relative seat order), tokens sorted by progress descending
    for k in range(1, 4):
        o = (pl + k) % 4
        progs = sorted(((x + 1) / 57 for x in pos[o]), reverse=True)
        for j, v in enumerate(progs):
            f[f"opp{k}_prog_{j}"] = round(v, 4)
    return f, infos


# ---------------------------------------------------------------- policies
def expert_scores(pos, pl, dice, infos, rng):
    """Rule-based expert: capture > finish > leave yard > escape > safe > advance."""
    scores = {}
    for t in range(4):
        i = infos[t]
        if not i["legal"]:
            continue
        p = pos[pl][t]
        s = 0.0
        if i["capture"]:
            s += 100 + 20 * i["cap_prog"]
        if i["finish"]:
            s += 90
        if i["leave"]:
            s += 70
        if i["danger_now"] > 0 and i["danger_after"] == 0:
            s += 50 * i["danger_now"] + 10          # escaping a threat
        if i["enter_home"]:
            s += 45
        if i["safe_after"] and not i["leave"]:
            s += 25
        s -= 35 * i["danger_after"]                 # moving into danger
        s += (max(p, 0) + dice) / 10                # prefer advancing the leader
        s += rng.random() * 0.01                    # random tie-break
        scores[t] = s
    return scores


def choose_move(policy, legal, expert_move, rng):
    kind, eps = policy
    if kind == "random":
        return rng.choice(legal)
    if rng.random() < eps:                          # exploration noise
        return rng.choice(legal)
    return expert_move


def apply_move(pos, pl, t, dice):
    np_ = new_progress(pos[pl][t], dice)
    caps = captured_at(pos, pl, abs_sq(pl, np_))
    pos[pl][t] = np_
    for o, tt in caps:
        pos[o][tt] = -1
    return bool(caps)


# ---------------------------------------------------------------- game loop
def play_game(rng, policies, keep_forced):
    pos = [[-1] * 4 for _ in range(4)]
    rows, turns, pl = [], 0, 0
    while turns < MAX_TURNS:
        sixes = 0
        while True:
            dice = rng.randint(1, 6)
            turns += 1
            if dice == 6:
                sixes += 1
                if sixes == 3:
                    break                           # third six: forfeit
            legal = [t for t in range(4) if is_legal(pos[pl][t], dice)]
            extra = dice == 6
            if legal:
                feats, infos = build_features(pos, pl, dice)
                scores = expert_scores(pos, pl, dice, infos, rng)
                expert_move = max(scores, key=scores.get)
                played = choose_move(policies[pl], legal, expert_move, rng)
                if len(legal) >= 2 or keep_forced:
                    row = {"turn": turns, "player": pl, "dice": dice,
                           "n_legal": len(legal), **feats,
                           "expert_action": expert_move, "played_action": played,
                           "played_by": f"{policies[pl][0]}_{policies[pl][1]}"}
                    rows.append(row)
                if apply_move(pos, pl, played, dice):
                    extra = True
                if all(x == FINISH for x in pos[pl]):
                    return rows, pl
            if not extra:
                break
        pl = (pl + 1) % 4
    return rows, None                               # timed out (very rare)


def random_policies(rng):
    options = [("expert", 0.0), ("expert", 0.1), ("expert", 0.2),
               ("expert", 0.3), ("random", 1.0)]
    return [rng.choice(options) for _ in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ludo_dataset.csv")
    ap.add_argument("--keep-forced", action="store_true",
                    help="also log decisions with only one legal move")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    writer, n_rows, skipped = None, 0, 0
    with open(args.out, "w", newline="") as fh:
        for g in range(args.games):
            rows, winner = play_game(rng, random_policies(rng), args.keep_forced)
            if winner is None:
                skipped += 1
                continue
            split = "train" if g % 10 < 8 else ("val" if g % 10 == 8 else "test")
            for r in rows:
                r = {"game_id": g, "split": split, **r,
                     "won": int(r["player"] == winner)}
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(r.keys()))
                    writer.writeheader()
                writer.writerow(r)
                n_rows += 1
            if (g + 1) % 500 == 0:
                print(f"{g + 1}/{args.games} games, {n_rows} rows")
    print(f"Done: {n_rows} rows from {args.games - skipped} games -> {args.out}")


if __name__ == "__main__":
    main()
