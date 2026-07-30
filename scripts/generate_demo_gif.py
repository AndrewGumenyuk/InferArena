"""Generate the InferArena demo GIF for the README.

Renders a terminal-style animation of `inferarena compare` running the
case-study workload with three schedulers, then saves
docs/assets/inferarena-demo.gif.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 960, 560
BG = (13, 17, 23)  # GitHub dark
FG = (201, 209, 217)
GREEN = (63, 185, 80)
BLUE = (88, 166, 255)
ORANGE = (210, 153, 34)
RED = (248, 81, 73)
DIM = (110, 118, 129)
FPS = 8
HOLD_LAST_FRAMES = FPS * 3  # hold the final frame ~3s

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT = ImageFont.truetype(FONT_PATH, 13)
FONT_BOLD = ImageFont.truetype(FONT_PATH, 13, index=1)

COMMAND = "inferarena compare --config examples/case_study_variable.yaml --schedulers fcfs,sjf,sarathi_serve"

# (text, color) lines printed after the command, revealed progressively.
OUTPUT_LINES: list[tuple[str, tuple[int, int, int]]] = [
    ("", FG),
    ("Running experiment: case-study-variable (64 requests, seed=42)", DIM),
    ("  [1/3] fcfs           ... done (20000 steps)", FG),
    ("  [2/3] sjf            ... done (20000 steps)", FG),
    ("  [3/3] sarathi_serve  ... done (20000 steps)", FG),
    ("", FG),
    ("scheduler        completed  throughput   ttft_p50  latency_p50", BLUE),
    ("fcfs                   2/64     0.09 rps     23.3ms     1457.9ms", RED),
    ("sjf                   33/64     0.93 rps     38.6ms     1595.2ms", ORANGE),
    ("sarathi_serve         64/64     1.03 rps  22789.0ms    24333.4ms", GREEN),
    ("", FG),
    ("Plot:    inferarena_outputs/case-study/comparison.png", DIM),
    ("Report:  inferarena_outputs/case-study/report.md", DIM),
    ("", FG),
    ("Sarathi-Serve completes all 64 requests (32x FCFS).", GREEN),
    ("Latency percentiles reflect completed requests only.", DIM),
    ("Details: docs/explanation/case-study.md", DIM),
]


def render_frame(typed: int, revealed: int, cursor_on: bool) -> Image.Image:
    """Render one frame with `typed` chars of the command and `revealed` output lines."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    # Window chrome
    draw.rounded_rectangle([8, 8, WIDTH - 8, HEIGHT - 8], radius=10, outline=(48, 54, 61))
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([24 + i * 24, 20, 36 + i * 24, 32], fill=color)

    y = 56
    prompt = "$ "
    draw.text((28, y), prompt, font=FONT_BOLD, fill=GREEN)
    visible = COMMAND[:typed]
    draw.text((28 + draw.textlength(prompt, font=FONT_BOLD), y), visible, font=FONT, fill=FG)
    if cursor_on and revealed == 0:
        x = 28 + draw.textlength(prompt + visible, font=FONT_BOLD)
        draw.rectangle([x, y + 2, x + 9, y + 18], fill=FG)
    y += 30

    for text, color in OUTPUT_LINES[:revealed]:
        draw.text((28, y), text, font=FONT, fill=color)
        y += 20
    if cursor_on and revealed > 0:
        draw.rectangle([28, y + 2, 37, y + 18], fill=FG)
    return img


def main() -> None:
    """Generate the GIF."""
    frames: list[Image.Image] = []
    durations: list[int] = []

    # Phase 1: type the command (2 chars per frame).
    for typed in range(0, len(COMMAND) + 1, 2):
        frames.append(render_frame(typed, 0, cursor_on=True))
        durations.append(1000 // FPS)

    # Pause at full command.
    for _ in range(FPS // 2):
        frames.append(render_frame(len(COMMAND), 0, cursor_on=True))
        durations.append(1000 // FPS)

    # Phase 2: reveal output lines, one every ~0.4s.
    for revealed in range(1, len(OUTPUT_LINES) + 1):
        frames.append(render_frame(len(COMMAND), revealed, cursor_on=True))
        durations.append(400)

    # Phase 3: hold the final frame, blinking cursor.
    for i in range(HOLD_LAST_FRAMES):
        frames.append(render_frame(len(COMMAND), len(OUTPUT_LINES), cursor_on=i % 2 == 0))
        durations.append(1000 // FPS)

    out = Path("docs/assets/inferarena-demo.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = out.stat().st_size / 1024
    total_s = sum(durations) / 1000
    print(f"Saved {out} ({size_kb:.0f} KB, {len(frames)} frames, {total_s:.1f}s)")


if __name__ == "__main__":
    main()
