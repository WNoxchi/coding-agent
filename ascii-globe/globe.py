#!/usr/bin/env python3
"""
ASCII Rotating Earth Globe
Displays a detailed rotating Earth rendered in ASCII characters with a space background.
"""

import curses
import math
import time
import random
import sys

# ── Earth texture ──────────────────────────────────────────────────────────────
# 36 rows × 72 columns mapping latitude/longitude to terrain type.
# Each character encodes:
#   '~' deep ocean      '.' shallow / coastal ocean
#   'o' lowland         '^' highlands / mountains
#   '#' dense land      ' ' open ocean (space)
# We store the map as a flat string indexed by [row * 72 + col].
# Latitude rows run from +90° (top) to -90° (bottom).
# Longitude columns run from -180° (left) to +180° (right).

EARTH_ROWS = 36
EARTH_COLS = 72

# fmt: off
EARTH_MAP = (
    # Row  0  (+90° → ~80°N)  Arctic ocean / ice cap
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # Row  1  (~80°N)
    "~~~~^^^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # Row  2  (~75°N)  Greenland / northern Canada / Siberia
    "~~~~###^~~~~~~~~~~~~~~~~~~~~^####~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~"
    # Row  3  (~70°N)
    "~~~^####~~~~~~~~~~~~~~~~~~~~######^~~~~~~~~~~~~~~~~~~~~~~~~~##~~~~~~~~~~"
    # Row  4  (~65°N)  Alaska / Scandinavia / Russia
    "~~^#####^~~~~~~~~~~~~~~~~~~^######~~~^^^^^~~~~~~~~~~~~~~~~~~###^~~~~~~~~"
    # Row  5  (~60°N)  Canada / N. Europe / Russia
    "~^######^~~~~~~~~~~~~~~~~~^########^^#####^~~~~~~~~~~~~~~~~^####^~~~~~~~"
    # Row  6  (~55°N)
    "~~^#####~~~~~~~~~~~~~~~~~~~^#############^~~~~~~~~~~~~~~~~~~####^~~~~~~~"
    # Row  7  (~50°N)  UK / W. Europe / C. Asia
    "~~~^###^~~~~^^^^^~~~~~~~~~~~^############^~~~~~~~~~~~~~~~~~~^###^~~~~~~~"
    # Row  8  (~45°N)  France / C. Europe / Central Asia
    "~~~~~~~~~^^######^^^~~~~~~~~~^###########^~~~~~~~~~~~~~~~~~~^####~~~~~~~"
    # Row  9  (~40°N)  Spain / Turkey / China / Japan
    "~~~~~~~~^########^~~~~~~~~~~~~^#######^~~~~~~~~~~~~~~~~~~~~^^####^~~~~~~"
    # Row 10  (~35°N)  N. Africa / Middle East / S. China
    "~~~~~~~^##^^^^^###^~~~~~~~~~~~~^###^~~~~~~~~~~~~~~~~~~~~~^^^#####~~~~~~~"
    # Row 11  (~30°N)  Sahara / Arabia / India
    "~~~~~~~####.....###^~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~^^#########~~~~~~"
    # Row 12  (~25°N)  Sahara / India / SE Asia
    "~~~~~~^###......^##^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^####~~~~~~~"
    # Row 13  (~20°N)  Sahara / Tropics
    "~~~~~~^##^......^#^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~..~~~~^####~~~~~~"
    # Row 14  (~15°N)  W. Africa / Horn / SE Asia
    "~~~~~~^##^.....^##^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~...~~~~^###^~~~~~~"
    # Row 15  (~10°N)  C. Africa / India tip / SE Asia
    "~~~~~~~^##^....###^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~....^~~~^##^~~~~~~~"
    # Row 16  (~5°N )  C. Africa / Indonesia
    "~~~~~~~^###^..^###^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~.....^^^^###^~~~~~~~"
    # Row 17  ( 0°  )  Equator  Congo / Borneo / S. America north
    "^~~~~~~^####^^####^~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^.....^#####^~~~~~~~~"
    # Row 18  (~5°S )  S. America / C. Africa / Indonesia
    "^#^~~~~^###########^~~~~~~~~~~~~~~~~~~~~~~~~^#####^.....^######^~~~~~~~~"
    # Row 19  (~10°S)  Brazil / S. Africa
    "^##^~~~^############^~~~~~~~~~~~~~~~~~~~~~~~^######....^######^~~~~~~~~~"
    # Row 20  (~15°S)  Brazil / Angola
    "^##^~~~~^###########^~~~~~~~~~~~~~~~~~~~~~~~~^#####^..^#####^~~~~~~~~~~"
    # Row 21  (~20°S)  Brazil / Mozambique
    "~^##^~~~~^#########^~~~~~~~~~~~~~~~~~~~~~~~~~~^####^^^####^~~~~~~~~~~~~"
    # Row 22  (~25°S)  SE Brazil / S. Africa
    "~~^##^~~~~^#######^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^#######^~~~~~~~~~~~~~~"
    # Row 23  (~30°S)
    "~~~^##^~~~~^#####^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^###^~~~~~~~~~~~~~~~"
    # Row 24  (~35°S)  Argentina / S. Africa tip
    "~~~~^##^~~~~^###^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^#^~~~~~~~~~~~~~~~~"
    # Row 25  (~40°S)  Argentina / New Zealand
    "~~~~~^##^~~~~^#^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^"
    # Row 26  (~45°S)
    "~~~~~~^##^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^##^"
    # Row 27  (~50°S)  Patagonia / open Southern Ocean
    "~~~~~~~^#^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # Row 28  (~55°S)  Tip of S. America / Drake Passage
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # Row 29  (~60°S)  Southern Ocean
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    # Row 30  (~65°S)  Antarctica coast
    "~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^~~~~~~~~~~~~~~~~"
    # Row 31  (~70°S)  Antarctica
    "~~~~~~~~~~~~~~~~~~~~~~^###################################^~~~~~~~~~~~~~"
    # Row 32  (~75°S)
    "~~~~~~~~~~~~~~~~~~~~~^#####################################^~~~~~~~~~~~~"
    # Row 33  (~80°S)
    "~~~~~~~~~~~~~~~~~~~~~^######################################^~~~~~~~~~~~"
    # Row 34  (~85°S)
    "~~~~~~~~~~~~~~~~~~~~~~^#####################################^~~~~~~~~~~~"
    # Row 35  (~90°S)  South Pole
    "~~~~~~~~~~~~~~~~~~~~~~~~#################################~~~~~~~~~~~~~~~~"
)
# fmt: on

# Pad / truncate every row to exactly EARTH_COLS characters
_rows = []
for i in range(EARTH_ROWS):
    row = EARTH_MAP[i * EARTH_COLS : (i + 1) * EARTH_COLS]
    _rows.append(row.ljust(EARTH_COLS)[:EARTH_COLS])
EARTH_MAP = _rows   # list of 36 strings, each 72 chars


def sample_earth(lat_deg: float, lon_deg: float) -> str:
    """Return the map character for a given latitude/longitude in degrees."""
    row = int((90.0 - lat_deg) / 180.0 * EARTH_ROWS)
    col = int((lon_deg + 180.0) / 360.0 * EARTH_COLS)
    row = max(0, min(EARTH_ROWS - 1, row))
    col = col % EARTH_COLS
    return EARTH_MAP[row][col]


# ── Rendering helpers ──────────────────────────────────────────────────────────

# ASCII shading palette from dark→bright (used for lighting the globe)
SHADING = " .,:;+*?%#@"

def terrain_shade(char: str, brightness: float) -> str:
    """Map a terrain char + brightness [0,1] to an ASCII shade character."""
    if char in (' ', '~'):
        # Ocean: use blue-ish shading characters
        shades = " ·.~≈~.·"
        idx = int(brightness * (len(shades) - 1))
        return shades[max(0, min(len(shades) - 1, idx))]
    else:
        # Land: use denser characters
        shades = ".,:;oO0#@"
        idx = int(brightness * (len(shades) - 1))
        return shades[max(0, min(len(shades) - 1, idx))]


def is_ocean(char: str) -> bool:
    return char in (' ', '~', '.')


# ── Globe projection ───────────────────────────────────────────────────────────

def render_globe(cx: float, cy: float, radius: float,
                 rotation: float, screen_rows: int, screen_cols: int):
    """
    Return a list of (row, col, char, color_pair) tuples for every pixel of
    the globe that falls within the screen.

    cx, cy    – centre of globe in screen coordinates
    radius    – radius in character-cells (we compensate for aspect ratio)
    rotation  – longitude offset in radians (increases over time)
    """
    pixels = []
    # Sun direction (fixed light source, slightly above-right)
    sun = (0.6, -0.4, 0.7)
    sun_len = math.sqrt(sum(v * v for v in sun))
    sun = tuple(v / sun_len for v in sun)

    # Terminal cells are roughly twice as tall as wide → scale y by 0.5
    ASPECT = 0.5

    for py in range(screen_rows):
        dy = (py - cy) / (radius * ASPECT)   # normalise to [-1,1] range
        if abs(dy) > 1.0:
            continue
        for px in range(screen_cols):
            dx = (px - cx) / radius
            d2 = dx * dx + dy * dy
            if d2 > 1.0:
                continue
            # Surface normal in 3-D (sphere of unit radius)
            dz = math.sqrt(max(0.0, 1.0 - d2))
            nx, ny, nz = dx, dy, dz

            # Convert normal back to lat/lon, then apply rotation
            lat = math.degrees(math.asin(max(-1.0, min(1.0, -ny))))
            lon = math.degrees(math.atan2(nx, nz)) + math.degrees(rotation)
            lon = ((lon + 180.0) % 360.0) - 180.0

            terrain = sample_earth(lat, lon)

            # Diffuse lighting (Lambertian)
            dot = nx * sun[0] + ny * sun[1] + nz * sun[2]
            brightness = max(0.05, dot)  # small ambient keeps dark side visible

            # Specular highlight on ocean
            if is_ocean(terrain):
                spec = max(0.0, dot) ** 8
                brightness = min(1.0, brightness + 0.3 * spec)

            ch = terrain_shade(terrain, brightness)

            # Colour pair selection
            if is_ocean(terrain):
                if brightness > 0.7:
                    color = 5   # bright ocean (cyan-ish)
                else:
                    color = 3   # dark ocean (blue)
            else:
                if terrain in ('^', '#'):
                    color = 6   # mountains (white/grey)
                elif brightness > 0.6:
                    color = 4   # bright land (green)
                else:
                    color = 2   # dark land (darker green)

            # Night side tint
            if dot < 0:
                color = 7   # very dim blue for night side

            pixels.append((py, px, ch, color))
    return pixels


# ── Star / comet / space background ───────────────────────────────────────────

class Star:
    CHARS = ['·', '✦', '+', '✧', '⋆', '*', '᛫']

    def __init__(self, rows, cols):
        self.reset(rows, cols, initial=True)

    def reset(self, rows, cols, initial=False):
        self.row = random.randint(2, rows - 3)
        self.col = random.randint(0, cols - 1)
        self.char = random.choice(self.CHARS)
        self.color = random.choice([8, 9, 10])   # dim white / dim cyan / dim yellow
        self.twinkle_phase = random.uniform(0, math.pi * 2)
        self.twinkle_speed = random.uniform(0.5, 2.0)
        if initial:
            self.age = random.uniform(0, 100)
        else:
            self.age = 0
        self.max_age = random.uniform(60, 200)

    def update(self, dt):
        self.age += dt * 10

    def is_dead(self):
        return self.age > self.max_age

    def visible(self, t):
        # Twinkle: sine wave gating
        return math.sin(t * self.twinkle_speed + self.twinkle_phase) > -0.5


class Comet:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.reset()

    def reset(self):
        # Start from a random edge
        edge = random.choice(['top', 'left', 'right'])
        if edge == 'top':
            self.row = 2.0
            self.col = random.uniform(0, self.cols)
        elif edge == 'left':
            self.row = random.uniform(2, self.rows - 2)
            self.col = 0.0
        else:
            self.row = random.uniform(2, self.rows - 2)
            self.col = float(self.cols - 1)
        # Direction: always roughly inward / across
        angle = random.uniform(math.pi / 8, math.pi * 3 / 8)
        if self.col > self.cols / 2:
            angle = math.pi - angle
        speed = random.uniform(8, 20)
        self.dr = math.sin(angle) * speed
        self.dc = math.cos(angle) * speed
        self.tail_len = random.randint(4, 10)
        self.trail = []   # list of (row, col) past positions
        self.alive = True
        self.color = random.choice([11, 12, 13])   # bright white / cyan / yellow

    def update(self, dt):
        self.trail.append((self.row, self.col))
        if len(self.trail) > self.tail_len:
            self.trail.pop(0)
        self.row += self.dr * dt
        self.col += self.dc * dt
        if (self.row < 2 or self.row >= self.rows - 2
                or self.col < 0 or self.col >= self.cols):
            self.alive = False

    def cells(self):
        """Yield (row, col, char, color, dim_factor) for head + tail."""
        # Head
        yield (int(self.row), int(self.col), '★', self.color, 1.0)
        # Tail (fading)
        for i, (tr, tc) in enumerate(reversed(self.trail)):
            fade = (i + 1) / (self.tail_len + 1)
            tail_chars = ['·', '·', '.', '.', ' ']
            ci = min(int(fade * len(tail_chars)), len(tail_chars) - 1)
            yield (int(tr), int(tc), tail_chars[ci], self.color, 1.0 - fade)


class Planet:
    """A tiny decorative background 'planet' (rare)."""
    GLYPHS = ['◉', '●', 'O', 'o']

    def __init__(self, rows, cols):
        self.row = random.randint(3, rows - 4)
        self.col = random.randint(1, cols - 2)
        self.glyph = random.choice(self.GLYPHS)
        self.color = random.choice([14, 15, 16])
        self.life = random.uniform(80, 200)
        self.age = 0.0
        # Slow drift
        self.dr = random.uniform(-0.2, 0.2)
        self.dc = random.uniform(-0.4, 0.4)
        self.frow = float(self.row)
        self.fcol = float(self.col)

    def update(self, dt):
        self.age += dt
        self.frow += self.dr * dt
        self.fcol += self.dc * dt
        self.row = int(self.frow)
        self.col = int(self.fcol)

    def is_dead(self, rows, cols):
        return (self.age > self.life
                or self.row < 2 or self.row >= rows - 2
                or self.col < 0 or self.col >= cols)


# ── Banner helpers ─────────────────────────────────────────────────────────────

def draw_banner(win, row: int, cols: int, text: str, color: int):
    """Draw a centred banner that fills the full width."""
    # Pad text to fill the line
    padded = text.center(cols - 1)
    try:
        win.attron(curses.color_pair(color) | curses.A_BOLD)
        win.addstr(row, 0, padded)
        win.attroff(curses.color_pair(color) | curses.A_BOLD)
    except curses.error:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

def init_colors():
    curses.start_color()
    curses.use_default_colors()

    # Globe colours  (pair index, fg, bg)
    curses.init_pair(2,  curses.COLOR_GREEN,   -1)   # dark land
    curses.init_pair(3,  curses.COLOR_BLUE,    -1)   # dark ocean
    curses.init_pair(4,  curses.COLOR_GREEN,   -1)   # bright land
    curses.init_pair(5,  curses.COLOR_CYAN,    -1)   # bright ocean
    curses.init_pair(6,  curses.COLOR_WHITE,   -1)   # mountains / ice
    curses.init_pair(7,  curses.COLOR_BLUE,    -1)   # night side
    # Star colours
    curses.init_pair(8,  curses.COLOR_WHITE,   -1)   # dim star
    curses.init_pair(9,  curses.COLOR_CYAN,    -1)   # dim cyan star
    curses.init_pair(10, curses.COLOR_YELLOW,  -1)   # dim yellow star
    # Comet colours
    curses.init_pair(11, curses.COLOR_WHITE,   -1)
    curses.init_pair(12, curses.COLOR_CYAN,    -1)
    curses.init_pair(13, curses.COLOR_YELLOW,  -1)
    # Planet colours
    curses.init_pair(14, curses.COLOR_RED,     -1)
    curses.init_pair(15, curses.COLOR_MAGENTA, -1)
    curses.init_pair(16, curses.COLOR_YELLOW,  -1)
    # Banner colours
    curses.init_pair(17, curses.COLOR_BLACK,   curses.COLOR_CYAN)    # Hello World!
    curses.init_pair(18, curses.COLOR_BLACK,   curses.COLOR_GREEN)   # Built with Coder!


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(33)   # ~30 fps

    init_colors()

    rows, cols = stdscr.getmaxyx()

    # ── Space objects ──
    NUM_STARS = max(30, (rows * cols) // 80)
    stars = [Star(rows, cols) for _ in range(NUM_STARS)]
    comets = []
    planets = []

    comet_timer = random.uniform(3, 8)    # seconds until first comet
    planet_timer = random.uniform(10, 30)

    ROTATION_PERIOD = 60.0   # seconds per full rotation
    start_time = time.monotonic()
    last_time = start_time

    while True:
        now = time.monotonic()
        dt = now - last_time
        last_time = now
        elapsed = now - start_time

        # ── Input ──
        key = stdscr.getch()
        if key in (ord('q'), ord('Q'), 27):
            break

        # ── Recompute layout (terminal may resize) ──
        rows, cols = stdscr.getmaxyx()
        cx = cols / 2.0
        cy = rows / 2.0
        # Radius: fill most of the shorter dimension, leave room for banners
        usable_rows = rows - 4   # 2 banner rows top + bottom each + padding
        radius = min(cols / 2.5, usable_rows / 1.1) * 0.85
        radius = max(4.0, radius)

        rotation = (elapsed / ROTATION_PERIOD) * 2.0 * math.pi

        # ── Update space objects ──
        # Stars
        for s in stars:
            s.update(dt)
        stars = [s for s in stars if not s.is_dead()]
        while len(stars) < NUM_STARS:
            stars.append(Star(rows, cols))

        # Comets
        comet_timer -= dt
        if comet_timer <= 0:
            comets.append(Comet(rows, cols))
            comet_timer = random.uniform(4, 12)
        for c in comets:
            c.update(dt)
        comets = [c for c in comets if c.alive]

        # Planets
        planet_timer -= dt
        if planet_timer <= 0 and len(planets) < 2:
            planets.append(Planet(rows, cols))
            planet_timer = random.uniform(15, 40)
        for p in planets:
            p.update(dt)
        planets = [p for p in planets if not p.is_dead(rows, cols)]

        # ── Build frame ──
        stdscr.erase()

        # 1. Stars (draw first, behind everything)
        for s in stars:
            if 2 <= s.row < rows - 2 and 0 <= s.col < cols - 1:
                if s.visible(elapsed):
                    try:
                        attr = curses.color_pair(s.color) | curses.A_DIM
                        stdscr.addstr(s.row, s.col, s.char, attr)
                    except curses.error:
                        pass

        # 2. Planets (subtle, behind comets and globe)
        for p in planets:
            if 2 <= p.row < rows - 2 and 0 <= p.col < cols - 1:
                try:
                    attr = curses.color_pair(p.color) | curses.A_DIM
                    stdscr.addstr(p.row, p.col, p.glyph, attr)
                except curses.error:
                    pass

        # 3. Comets
        for c in comets:
            for cr, cc, ch, ccol, fade in c.cells():
                if 2 <= cr < rows - 2 and 0 <= cc < cols - 1:
                    try:
                        attr = curses.color_pair(ccol)
                        if fade < 0.5:
                            attr |= curses.A_DIM
                        else:
                            attr |= curses.A_BOLD
                        stdscr.addstr(cr, cc, ch, attr)
                    except curses.error:
                        pass

        # 4. Globe (rendered over the background)
        globe_pixels = render_globe(cx, cy, radius, rotation, rows, cols)
        for pr, pc, ch, color in globe_pixels:
            if 2 <= pr < rows - 2 and 0 <= pc < cols - 1:
                try:
                    # Choose bold for lit areas, dim for night
                    if color == 7:
                        attr = curses.color_pair(color) | curses.A_DIM
                    else:
                        attr = curses.color_pair(color) | curses.A_BOLD
                    stdscr.addstr(pr, pc, ch, attr)
                except curses.error:
                    pass

        # 5. Banners (drawn last so they are always on top)
        draw_banner(stdscr, 0,        cols, "✦  Hello World!  ✦",    17)
        draw_banner(stdscr, 1,        cols, "",                        17)
        draw_banner(stdscr, rows - 2, cols, "",                        18)
        draw_banner(stdscr, rows - 1, cols, "✦  Built with Coder!  ✦", 18)

        stdscr.refresh()

    curses.curs_set(1)


def run():
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure terminal is restored
        print("\033[?25h", end='', flush=True)   # show cursor


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('-h', '--help'):
        print("ASCII Globe  –  a rotating Earth in your terminal.")
        print("Controls: q / Q / Esc  →  quit")
        print("Best viewed in a terminal at least 80×24 characters.")
        sys.exit(0)
    run()
