"""
Play Ludo vs the ANN inside Google Colab / Jupyter (no window needed).

    from colab_play import play
    play()                 # you (Red) vs 3 ANN players
    play(opponents=1)      # heads-up

The board is drawn off-screen with pygame and shown as an image after every
turn. Type your answers into the input box: press Enter to roll, then type
the token number (1-4) to move. Type q to quit.

The game runs on a virtual clock (no real-time waiting), so PC turns resolve
instantly and timing can never make the game hang.
"""
import io
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import ludo_game as L  # noqa: E402

try:
    from IPython.display import Image, clear_output, display
except ImportError:                      # plain Python: save frames instead
    Image = display = None
    clear_output = lambda **k: None      # noqa: E731


class _VirtualClock:
    t = 0


def _show(ui, path="board.png"):
    ui.dice_flicker_until = 0
    ui.draw()
    buf = io.BytesIO()
    ui.pg.image.save(ui.screen, buf, "board.png")
    data = buf.getvalue()
    if display:
        clear_output(wait=True)
        display(Image(data=data))
    else:
        with open(path, "wb") as f:
            f.write(data)
    return data


def _settle(ui, clk, cap=3000):
    """Run PC turns/animations until it is the human's turn (or the game ends)."""
    n = 0
    while not ui.over and ui.mode not in ("roll", "pick"):
        clk.t += 50                      # advance virtual time
        ui.update()
        n += 1
        if n > cap:                      # should never happen; recover instead of hanging
            print("(recovering a stalled turn)")
            ui.pending = None
            ui.start_turn()
            n = 0


def play(opponents=3, brain="ANN", seed=None):
    ui = L.LudoUI(opponents=opponents, speed=1.0, seed=seed)
    clk = _VirtualClock()
    ui.now = lambda: clk.t               # replace the real clock
    ui.new_game()
    ui.set_brain(brain)
    while True:
        _settle(ui, clk)
        _show(ui)
        for text, _ in ui.log[-5:]:
            print(text)
        if ui.over:
            w = ui.g.winner
            print("\n*** YOU WIN! ***" if w == 0 else f"\n*** {ui.pname(w)} wins. ***")
            return
        if ui.mode == "roll":
            ans = input("Your turn - press Enter to roll (q to quit): ").strip().lower()
            if ans == "q":
                return
            ui.human_roll()
        else:  # pick
            opts = ", ".join(str(t + 1) for t in ui.legal)
            ans = input(f"Rolled {ui.g.dice}. Which token? [{opts}] (q to quit): ").strip().lower()
            if ans == "q":
                return
            if ans.isdigit() and int(ans) - 1 in ui.legal:
                ui.pick(int(ans) - 1)
            else:
                print("Not a legal choice, try again.")
