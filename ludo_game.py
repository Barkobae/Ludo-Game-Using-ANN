#!/usr/bin/env python3
"""
Ludo: you (Red) vs PC opponents controlled by the trained ANN.

    pip install pygame numpy pandas
    python train_ann.py            # once (a pre-trained ludo_ann.npz is included)
    python ludo_game.py            # you vs 3 ANN players
    python ludo_game.py --opponents 1     # heads-up: you vs 1 ANN player
    python ludo_game.py --demo            # watch the ANNs play each other
    python ludo_game.py --selftest 500    # headless strength test (no window)

Controls
    SPACE / ENTER / click ROLL   roll the dice
    click a glowing token  (or press 1-4)   choose which token to move
    A / E / R                    PC brain: ANN / rule-based Expert / Random
    N                            new game          ESC  quit
"""
import argparse
import math
import os
import random
import subprocess
import sys

import numpy as np

from ludo_dataset_generator import (FINISH, N, SAFE, START, abs_sq, build_features,
                                    captured_at, expert_scores, is_legal, new_progress)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "ludo_ann.npz")
NAMES = ["Red", "Green", "Yellow", "Blue"]


# =====================================================================
#  Board geometry (15x15 grid). Track index i -> grid cell.
# =====================================================================
def _build_path():
    c = []
    c += [(x, 6) for x in range(1, 6)]
    c += [(6, y) for y in range(5, -1, -1)]
    c += [(7, 0), (8, 0)]
    c += [(8, y) for y in range(1, 6)]
    c += [(x, 6) for x in range(9, 15)]
    c += [(14, 7), (14, 8)]
    c += [(x, 8) for x in range(13, 8, -1)]
    c += [(8, y) for y in range(9, 15)]
    c += [(7, 14), (6, 14)]
    c += [(6, y) for y in range(13, 8, -1)]
    c += [(x, 8) for x in range(5, -1, -1)]
    c += [(0, 7), (0, 6)]
    return c


def rotcell(c, k):
    x, y = c
    for _ in range(k % 4):
        x, y = 14 - y, x
    return x, y


def rotc(pt, k):
    X, Y = pt
    for _ in range(k % 4):
        X, Y = 15 - Y, X
    return X, Y


PATH = _build_path()
assert len(PATH) == N == len(set(PATH))
assert all(rotcell(PATH[i], 1) == PATH[(i + 13) % N] for i in range(N)), "board must be 4-fold symmetric"

YARD_SPOTS = [(2, 2), (4, 2), (2, 4), (4, 4)]


def cell_pos(pl, p, t):
    """Position of token t of player pl at progress p, in continuous cell units."""
    if p == -1:
        return rotc(YARD_SPOTS[t], pl)
    if p <= 50:
        x, y = PATH[(START[pl] + p) % N]
        return x + 0.5, y + 0.5
    if p <= 55:
        x, y = rotcell((1 + (p - 51), 7), pl)
        return x + 0.5, y + 0.5
    return rotc((6.7, 7.5 + (t - 1.5) * 0.34), pl)


# =====================================================================
#  Game logic (no graphics)
# =====================================================================
class LudoGame:
    def __init__(self, seats, rng):
        self.seats = seats
        self.rng = rng
        self.pos = [[-1] * 4 for _ in range(4)]
        self.cur = seats[0]
        self.sixes = 0
        self.dice = None
        self.winner = None

    def end_turn(self):
        self.sixes = 0
        self.cur = self.seats[(self.seats.index(self.cur) + 1) % len(self.seats)]

    def roll(self):
        d = self.rng.randint(1, 6)
        self.dice = d
        out = dict(pl=self.cur, dice=d, legal=[], forfeit=False)
        if d == 6:
            self.sixes += 1
            if self.sixes == 3:                       # third six in a row: forfeit
                out["forfeit"] = True
                self.end_turn()
                return out
        legal = [t for t in range(4) if is_legal(self.pos[self.cur][t], d)]
        out["legal"] = legal
        if not legal and d != 6:
            self.end_turn()
        return out

    def move(self, t):
        pl, d = self.cur, self.dice
        frm = self.pos[pl][t]
        to = new_progress(frm, d)
        caps = captured_at(self.pos, pl, abs_sq(pl, to))
        self.pos[pl][t] = to
        old = {}
        for o, tt in caps:
            old[(o, tt)] = self.pos[o][tt]
            self.pos[o][tt] = -1
        extra = d == 6 or bool(caps)
        won = all(x == FINISH for x in self.pos[pl])
        if won:
            self.winner = pl
        elif not extra:
            self.end_turn()
        return dict(pl=pl, t=t, frm=frm, to=to, caps=caps, old=old, extra=extra, won=won)


# =====================================================================
#  Brains: each is callable(pos, pl, dice, legal) -> token index
# =====================================================================
class ANNBrain:
    name = "ANN"

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cols = [str(c) for c in z["feature_cols"]]
        n = sum(1 for k in z.files if k.startswith("W"))
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        self.last_probs = None

    def probs(self, pos, pl, dice, legal):
        feats, _ = build_features(pos, pl, dice)
        h = np.array([feats[c] for c in self.cols])
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        mask = np.zeros(4)
        mask[legal] = 1
        z = np.where(mask > 0, h, -1e9)
        e = np.exp(z - z.max()) * mask
        return e / e.sum()

    def __call__(self, pos, pl, dice, legal):
        self.last_probs = self.probs(pos, pl, dice, legal)
        return int(self.last_probs.argmax())


class ExpertBrain:
    name = "Expert"

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, pos, pl, dice, legal):
        _, infos = build_features(pos, pl, dice)
        s = expert_scores(pos, pl, dice, infos, self.rng)
        return max(s, key=s.get)


class RandomBrain:
    name = "Random"

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, pos, pl, dice, legal):
        return self.rng.choice(legal)


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    print("ludo_ann.npz not found - training it now from ludo_dataset_sample.csv ...")
    subprocess.check_call([sys.executable, os.path.join(HERE, "train_ann.py")], cwd=HERE)


# =====================================================================
#  Headless strength test
# =====================================================================
def play_headless(brains, seats, rng, max_rolls=20000):
    g = LudoGame(seats, rng)
    for _ in range(max_rolls):
        r = g.roll()
        if r["legal"]:
            g.move(brains[g.cur](g.pos, g.cur, g.dice, r["legal"]))
            if g.winner is not None:
                return g.winner
    return None


def selftest(n_games, seed=0):
    rng = random.Random(seed)
    ann = ANNBrain(MODEL_PATH)
    rnd, exp = RandomBrain(rng), ExpertBrain(rng)
    setups = [
        ("ANN vs 3 Random", ann, rnd, [0, 1, 2, 3]),
        ("ANN vs 3 Expert", ann, exp, [0, 1, 2, 3]),
        ("ANN vs 1 Random (heads-up)", ann, rnd, [0, 2]),
        ("Expert vs 3 Random (reference)", exp, rnd, [0, 1, 2, 3]),
    ]
    print(f"{'Match-up':36s} {'win rate':>9s}   (fair share)")
    for label, subj, opp, seats in setups:
        wins = 0
        for _ in range(n_games):
            me = rng.choice(seats)                     # random seat removes seat bias
            brains = {s: (subj if s == me else opp) for s in seats}
            wins += play_headless(brains, seats, rng) == me
        print(f"{label:36s} {wins / n_games:9.1%}   ({1 / len(seats):.0%})")


# =====================================================================
#  Pygame UI
# =====================================================================
COLORS = [(214, 62, 62), (52, 160, 84), (238, 190, 30), (56, 108, 214)]
TINTS = [(248, 214, 214), (206, 238, 214), (252, 240, 196), (208, 222, 248)]
DARK = [(150, 36, 36), (30, 110, 56), (170, 130, 10), (30, 70, 150)]
BG = (34, 38, 48)
PANEL = (44, 49, 62)
TEXT = (232, 234, 240)
MUTED = (150, 156, 172)

CELL, X0, Y0 = 44, 20, 20
BOARD_PX = 15 * CELL
WIN_W, WIN_H = X0 * 2 + BOARD_PX + 290, Y0 * 2 + BOARD_PX
PX = X0 * 2 + BOARD_PX                                  # panel left edge
ROLL_BTN = None


def px(pt):
    return X0 + pt[0] * CELL, Y0 + pt[1] * CELL


class LudoUI:
    STEP_MS = 120

    def __init__(self, opponents=3, demo=False, speed=1.0, seed=None):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Ludo - You vs ANN")
        self.clock = pygame.time.Clock()
        self.f_big = pygame.font.Font(None, 44)
        self.f_med = pygame.font.Font(None, 28)
        self.f_sm = pygame.font.Font(None, 22)
        self.f_tok = pygame.font.Font(None, 22)

        self.rng = random.Random(seed)
        ensure_model()
        self.ann = ANNBrain(MODEL_PATH)
        self.brains = {"ANN": self.ann, "Expert": ExpertBrain(self.rng), "Random": RandomBrain(self.rng)}
        self.brain_name = "ANN"
        self.opponents = opponents
        self.demo = demo
        self.speed = speed
        self.roll_rect = pygame.Rect(PX + 30, 262, 230, 52)
        self.new_game()

    # ---------------- setup / helpers ----------------
    def new_game(self):
        seats = [0, 1, 2, 3] if self.opponents == 3 else [0, 2]
        self.g = LudoGame(seats, self.rng)
        self.mode = "wait"            # 'roll' (human may roll), 'pick' (human picks token), 'wait'
        self.legal = []
        self.anim = None
        self.ghost = {}
        self.pending = None
        self.log = []
        self.dice_shown = None
        self.dice_flicker_until = 0
        self.ann_note = ""
        self.over = False
        self.say("New game. Roll a 6 to bring a token out!", MUTED)
        self.start_turn()

    def is_human(self, pl):
        return pl == 0 and not self.demo

    def pname(self, pl):
        if self.demo:
            return NAMES[pl]
        return "You" if pl == 0 else f"{NAMES[pl]} (PC)"

    def say(self, text, color=TEXT):
        self.log.append((text, color))
        self.log = self.log[-7:]

    def now(self):
        return self.pg.time.get_ticks()

    def schedule(self, delay_ms, fn):
        self.pending = (self.now() + delay_ms / self.speed, fn)

    # ---------------- turn flow ----------------
    def start_turn(self):
        if self.g.winner is not None:
            return
        if self.is_human(self.g.cur):
            self.mode = "roll"
        else:
            self.mode = "wait"
            self.schedule(700, self.roll_now)

    def human_roll(self):
        if self.mode == "roll" and self.g.winner is None:
            self.mode = "wait"
            self.roll_now()

    def roll_now(self):
        pl = self.g.cur
        r = self.g.roll()
        self.dice_shown = r["dice"]
        self.dice_flicker_until = self.now() + 350 / self.speed
        who = self.pname(pl)
        col = COLORS[pl]
        verb = "roll" if who == "You" else "rolls"
        if r["forfeit"]:
            self.say(f"{who} {verb} a third 6 - turn forfeited!", col)
            self.schedule(1100, self.start_turn)
        elif not r["legal"]:
            self.say(f"{who} {verb} {r['dice']} - no move possible.", col)
            self.schedule(1100, self.start_turn)
        elif self.is_human(pl):
            self.legal = r["legal"]
            if len(r["legal"]) == 1:
                self.say(f"You rolled {r['dice']} - only one move.", col)
                self.mode = "wait"
                self.schedule(650, lambda: self.do_move(r["legal"][0]))
            else:
                self.say(f"You rolled {r['dice']} - pick a token.", col)
                self.mode = "pick"
        else:
            brain = self.brains[self.brain_name]
            t = brain(self.g.pos, pl, r["dice"], r["legal"])
            if brain is self.ann and self.ann.last_probs is not None:
                pr = self.ann.last_probs
                self.ann_note = "ANN: " + "  ".join(f"T{i + 1} {pr[i]:.0%}" for i in r["legal"])
            else:
                self.ann_note = f"{self.brain_name} brain"
            self.say(f"{who} {verb} {r['dice']}.", col)
            self.legal = r["legal"]
            self.schedule(850, lambda: self.do_move(t))

    def pick(self, t):
        if self.mode == "pick" and t in self.legal:
            self.mode = "wait"
            self.do_move(t)

    def do_move(self, t):
        m = self.g.move(t)
        pl = m["pl"]
        seq = [-1, 0] if m["frm"] == -1 else list(range(m["frm"], m["to"] + 1))
        self.anim = dict(pl=pl, t=t, seq=seq, t0=self.now())
        self.ghost = dict(m["old"])
        dur = (len(seq) - 1) * self.STEP_MS + 150
        self.anim_end = self.now() + dur / self.speed
        self.legal = []
        who = self.pname(pl)
        txt = f"{who} moved token {t + 1}"
        if m["frm"] == -1:
            txt = f"{who} brought token {t + 1} out"
        if m["to"] == FINISH:
            txt = f"{who} got token {t + 1} HOME!"
        self.say(txt, COLORS[pl])
        for (o, tt) in m["caps"]:
            self.say(f"  captured {self.pname(o)}'s token {tt + 1}!", (255, 120, 120))
        if m["extra"] and not m["won"]:
            self.say(f"  {who} {'get' if who == 'You' else 'gets'} another turn.", MUTED)
        if m["won"]:
            self.schedule(dur + 400, self.game_over)
        else:
            self.schedule(dur + 450, self.start_turn)

    def game_over(self):
        self.over = True
        self.mode = "wait"

    # ---------------- per-frame logic ----------------
    def update(self):
        now = self.now()
        if self.anim and now >= self.anim_end:
            self.anim, self.ghost = None, {}
        if self.pending and now >= self.pending[0]:
            fn = self.pending[1]
            self.pending = None
            try:
                fn()
            except Exception:                      # never freeze silently: show the error, then recover
                import traceback
                traceback.print_exc()
                self.say("Internal error (see console) - recovering...", (255, 120, 120))
                self.mode, self.anim, self.ghost = "wait", None, {}
                self.schedule(800, self.start_turn)
        elif (self.pending is None and self.mode == "wait" and self.anim is None
              and not self.over and self.g.winner is None):
            self.start_turn()                      # watchdog: the game must never sit idle
        if self.demo and self.mode == "roll":
            self.human_roll()

    def handle_event(self, e):
        pg = self.pg
        if e.type == pg.KEYDOWN:
            if e.key == pg.K_ESCAPE:
                return False
            if e.key in (pg.K_SPACE, pg.K_RETURN):
                self.human_roll()
            elif e.key == pg.K_n:
                self.new_game()
            elif e.key == pg.K_a:
                self.set_brain("ANN")
            elif e.key == pg.K_e:
                self.set_brain("Expert")
            elif e.key == pg.K_r:
                self.set_brain("Random")
            elif e.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                self.pick(e.key - pg.K_1)
        elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
            if self.roll_rect.collidepoint(e.pos):
                self.human_roll()
            elif self.mode == "pick":
                best, bd = None, 22
                pos = self.token_positions()
                for t in self.legal:
                    x, y, _ = pos[(0, t)]
                    d = math.hypot(e.pos[0] - x, e.pos[1] - y)
                    if d < bd:
                        best, bd = t, d
                if best is not None:
                    self.pick(best)
        elif e.type == pg.QUIT:
            return False
        return True

    def set_brain(self, name):
        self.brain_name = name
        self.say(f"PC brain set to {name}.", MUTED)

    # ---------------- drawing ----------------
    def token_positions(self):
        now = self.now()
        base = {}
        for pl in self.g.seats:
            for t in range(4):
                p = self.g.pos[pl][t]
                if (pl, t) in self.ghost:
                    p = self.ghost[(pl, t)]
                if self.anim and self.anim["pl"] == pl and self.anim["t"] == t:
                    seq = self.anim["seq"]
                    s = (now - self.anim["t0"]) * self.speed / self.STEP_MS
                    k = min(int(s), len(seq) - 1)
                    a = cell_pos(pl, seq[k], t)
                    if k < len(seq) - 1:
                        b, f = cell_pos(pl, seq[k + 1], t), s - k
                        a = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
                    base[(pl, t)] = [*px(a), 16]
                else:
                    base[(pl, t)] = [*px(cell_pos(pl, p, t)), 16]
        groups = {}
        for key, (x, y, r) in base.items():
            groups.setdefault((round(x), round(y)), []).append(key)
        for keys in groups.values():
            if len(keys) > 1:
                for i, key in enumerate(keys):
                    ang = 2 * math.pi * i / len(keys) + 0.6
                    base[key][0] += 10 * math.cos(ang)
                    base[key][1] += 10 * math.sin(ang)
                    base[key][2] = 13
        return {k: tuple(v) for k, v in base.items()}

    def draw_star(self, cx, cy, r, color):
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        self.pg.draw.polygon(self.screen, color, pts)

    def draw_board(self):
        pg, s = self.pg, self.screen
        pg.draw.rect(s, (252, 250, 244), (X0 - 6, Y0 - 6, BOARD_PX + 12, BOARD_PX + 12), border_radius=10)
        for pl in range(4):                                         # yards
            (ax, ay), (bx, by) = rotc((0, 0), pl), rotc((6, 6), pl)
            pg.draw.rect(s, COLORS[pl], (*px((min(ax, bx), min(ay, by))), 6 * CELL, 6 * CELL))
            (ax, ay), (bx, by) = rotc((1, 1), pl), rotc((5, 5), pl)
            pg.draw.rect(s, TINTS[pl], (*px((min(ax, bx), min(ay, by))), 4 * CELL, 4 * CELL), border_radius=14)
            for t in range(4):
                x, y = px(rotc(YARD_SPOTS[t], pl))
                pg.draw.circle(s, COLORS[pl], (x, y), 19)
                pg.draw.circle(s, (255, 255, 255), (x, y), 15)
        for i, (x, y) in enumerate(PATH):                           # track
            col = (255, 255, 255)
            for pl in range(4):
                if i == START[pl]:
                    col = COLORS[pl]
            pg.draw.rect(s, col, (*px((x, y)), CELL, CELL))
            pg.draw.rect(s, (120, 124, 136), (*px((x, y)), CELL, CELL), 1)
            if i in SAFE and i not in START:
                cx, cy = px((x + .5, y + .5))
                self.draw_star(cx, cy, 14, (150, 154, 168))
        for pl in range(4):                                         # home columns
            for k in range(5):
                x, y = rotcell((1 + k, 7), pl)
                pg.draw.rect(s, COLORS[pl], (*px((x, y)), CELL, CELL))
                pg.draw.rect(s, (255, 255, 255), (*px((x, y)), CELL, CELL), 1)
        tri = [((6, 6), (6, 9)), ((6, 6), (9, 6)), ((9, 6), (9, 9)), ((6, 9), (9, 9))]
        for pl in range(4):                                         # centre triangles
            a, b = tri[pl]
            pg.draw.polygon(s, COLORS[pl], [px(a), px(b), px((7.5, 7.5))])
        for pl in range(4):                                         # start-square stars
            x, y = PATH[START[pl]]
            self.draw_star(*px((x + .5, y + .5)), 12, (255, 255, 255))

    def draw_dice(self):
        pg, s = self.pg, self.screen
        r = pg.Rect(PX + 100, 150, 90, 90)
        pg.draw.rect(s, (250, 250, 250), r, border_radius=14)
        pg.draw.rect(s, (90, 94, 108), r, 3, border_radius=14)
        v = self.dice_shown
        if self.now() < self.dice_flicker_until:
            v = random.randint(1, 6)
        if v is None:
            return
        grid = {1: [(1, 1)], 2: [(0, 0), (2, 2)], 3: [(0, 0), (1, 1), (2, 2)],
                4: [(0, 0), (2, 0), (0, 2), (2, 2)], 5: [(0, 0), (2, 0), (1, 1), (0, 2), (2, 2)],
                6: [(0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)]}[v]
        for gx, gy in grid:
            pg.draw.circle(s, (30, 30, 36), (r.x + 20 + gx * 25, r.y + 20 + gy * 25), 7)

    def draw_tokens(self):
        pg, s = self.pg, self.screen
        pos = self.token_positions()
        pulse = 3 * math.sin(self.now() / 140)
        order = sorted(pos.items(), key=lambda kv: kv[1][1])
        for (pl, t), (x, y, r) in order:
            if self.mode == "pick" and pl == 0 and t in self.legal:
                pg.draw.circle(s, (255, 255, 120), (x, y), r + 6 + pulse, 4)
            pg.draw.circle(s, (0, 0, 0), (x + 1, y + 3), r)
            pg.draw.circle(s, (255, 255, 255), (x, y), r)
            pg.draw.circle(s, COLORS[pl], (x, y), r - 3)
            pg.draw.circle(s, DARK[pl], (x, y), r - 3, 2)
            lab = self.f_tok.render(str(t + 1), True, (255, 255, 255))
            s.blit(lab, lab.get_rect(center=(x, y)))

    def draw_panel(self):
        pg, s = self.pg, self.screen
        pg.draw.rect(s, PANEL, (PX - 10, 0, WIN_W - PX + 10, WIN_H))
        t = self.f_big.render("LUDO", True, TEXT)
        s.blit(t, (PX + 30, 24))
        sub = self.f_sm.render("You vs ANN" if not self.demo else "ANN demo mode", True, MUTED)
        s.blit(sub, (PX + 30, 62))
        cur = self.g.cur
        pg.draw.rect(s, COLORS[cur], (PX + 30, 98, 230, 34), border_radius=8)
        who = "Game over" if self.g.winner is not None else f"{self.pname(cur)}'s turn"
        lab = self.f_med.render(who, True, (255, 255, 255))
        s.blit(lab, lab.get_rect(center=(PX + 145, 115)))
        self.draw_dice()
        active = self.mode == "roll"
        pg.draw.rect(s, (70, 170, 100) if active else (70, 76, 92), self.roll_rect, border_radius=12)
        lab = self.f_med.render("ROLL  (space)", True, (255, 255, 255) if active else MUTED)
        s.blit(lab, lab.get_rect(center=self.roll_rect.center))
        y = 336
        for pl in self.g.seats:                                      # scoreboard
            home = sum(1 for p in self.g.pos[pl] if p == FINISH)
            pg.draw.circle(s, COLORS[pl], (PX + 42, y + 10), 9)
            s.blit(self.f_sm.render(f"{self.pname(pl)}", True, TEXT), (PX + 60, y + 2))
            s.blit(self.f_sm.render(f"home {home}/4", True, MUTED), (PX + 190, y + 2))
            y += 26
        y += 8
        s.blit(self.f_sm.render(f"PC brain: {self.brain_name}", True, TEXT), (PX + 18, y))
        s.blit(self.f_sm.render("keys: A=ANN  E=Expert  R=Random", True, MUTED), (PX + 18, y + 19))
        if self.ann_note:
            s.blit(self.f_sm.render(self.ann_note, True, (140, 190, 255)), (PX + 18, y + 40))
        y += 70
        for text, col in self.log:
            s.blit(self.f_sm.render(text, True, col), (PX + 18, y))
            y += 21
        s.blit(self.f_sm.render("N = new game    ESC = quit", True, MUTED), (PX + 18, WIN_H - 28))

    def draw(self):
        self.screen.fill(BG)
        self.draw_board()
        self.draw_tokens()
        self.draw_panel()
        if self.over:
            ov = self.pg.Surface((BOARD_PX, 140), self.pg.SRCALPHA)
            ov.fill((0, 0, 0, 190))
            self.screen.blit(ov, (X0, Y0 + BOARD_PX // 2 - 70))
            w = self.g.winner
            msg = "YOU WIN!" if self.is_human(w) else f"{self.pname(w)} wins!"
            t = self.f_big.render(msg, True, COLORS[w])
            self.screen.blit(t, t.get_rect(center=(X0 + BOARD_PX // 2, Y0 + BOARD_PX // 2 - 18)))
            t = self.f_med.render("Press N for a new game", True, TEXT)
            self.screen.blit(t, t.get_rect(center=(X0 + BOARD_PX // 2, Y0 + BOARD_PX // 2 + 28)))
        self.pg.display.flip()

    def run(self, max_frames=None, screenshot=None):
        frames, running = 0, True
        while running:
            for e in self.pg.event.get():
                running = self.handle_event(e) and running
            self.update()
            self.draw()
            self.clock.tick(60)
            frames += 1
            if max_frames and frames >= max_frames:
                break
        if screenshot:
            self.pg.image.save(self.screen, screenshot)
        self.pg.quit()


def main():
    ap = argparse.ArgumentParser(description="Ludo vs ANN")
    ap.add_argument("--opponents", type=int, choices=[1, 3], default=3)
    ap.add_argument("--demo", action="store_true", help="ANNs play each other")
    ap.add_argument("--speed", type=float, default=1.0, help="PC/animation speed multiplier")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", type=int, metavar="N", help="run N headless games per match-up and exit")
    ap.add_argument("--frames", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--screenshot", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.selftest:
        ensure_model()
        selftest(a.selftest, a.seed or 0)
        return
    LudoUI(a.opponents, a.demo, a.speed, a.seed).run(a.frames, a.screenshot)


if __name__ == "__main__":
    main()
