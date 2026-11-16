"""
Pathfinding Visualizer  (Python + tkinter, no external libraries)

Algorithms : A* Search, Dijkstra, Breadth-First Search, Depth-First Search
Controls   : Left-click / drag   -> draw with the selected tool (Wall / Weight / Erase)
             Right-click / drag  -> erase
             Drag green / red    -> move the start / end cell
Weights    : "Weight" cells cost 5 to enter instead of 1. Dijkstra and A* respect
             them (and route around them); BFS and DFS ignore them.

Run:  python pathfinding_visualizer.py
"""

import heapq
import random
import tkinter as tk
from collections import deque
from tkinter import ttk

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ROWS, COLS, CELL = 25, 40, 24
WEIGHT_COST = 5
INF = float("inf")

COLORS = {
    "bg": "#f4f6f8",
    "empty": "#ffffff",
    "grid": "#d5dbe0",
    "wall": "#2b2f36",
    "start": "#2ecc71",
    "end": "#e74c3c",
    "frontier": "#b9e4ff",
    "visit": "#5aa9e6",
    "visit_w": "#b07cc6",
    "weight": "#f0a93b",
    "path": "#ffd23f",
}


# ----------------------------------------------------------------------------
# Algorithms
# Each one is a generator that yields events the GUI turns into animation:
#   ("frontier", node)  node discovered / added to the open list
#   ("visit", node)     node taken out and explored
#   ("done", path)      finished; path is a list of nodes, or None if no path
# ----------------------------------------------------------------------------
def reconstruct(parent, node):
    path = []
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


def bfs(start, end, neighbors, cost):
    parent = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        yield ("visit", cur)
        if cur == end:
            yield ("done", reconstruct(parent, cur))
            return
        for n in neighbors(cur):
            if n not in parent:
                parent[n] = cur
                queue.append(n)
                yield ("frontier", n)
    yield ("done", None)


def dfs(start, end, neighbors, cost):
    parent = {start: None}
    stack = [start]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        yield ("visit", cur)
        if cur == end:
            yield ("done", reconstruct(parent, cur))
            return
        for n in neighbors(cur):
            if n not in seen:
                parent[n] = cur
                stack.append(n)
                yield ("frontier", n)
    yield ("done", None)


def dijkstra(start, end, neighbors, cost):
    dist = {start: 0}
    parent = {start: None}
    heap = [(0, 0, start)]
    tie = 1  # tie-breaker so tuples never compare nodes
    closed = set()
    while heap:
        d, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        yield ("visit", cur)
        if cur == end:
            yield ("done", reconstruct(parent, cur))
            return
        for n in neighbors(cur):
            nd = d + cost(n)
            if nd < dist.get(n, INF):
                dist[n] = nd
                parent[n] = cur
                heapq.heappush(heap, (nd, tie, n))
                tie += 1
                yield ("frontier", n)
    yield ("done", None)


def astar(start, end, neighbors, cost):
    def h(node):  # Manhattan distance (admissible: every step costs >= 1)
        return abs(node[0] - end[0]) + abs(node[1] - end[1])

    g = {start: 0}
    parent = {start: None}
    heap = [(h(start), h(start), 0, start)]
    tie = 1
    closed = set()
    while heap:
        _, _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        yield ("visit", cur)
        if cur == end:
            yield ("done", reconstruct(parent, cur))
            return
        for n in neighbors(cur):
            ng = g[cur] + cost(n)
            if ng < g.get(n, INF):
                g[n] = ng
                parent[n] = cur
                heapq.heappush(heap, (ng + h(n), h(n), tie, n))
                tie += 1
                yield ("frontier", n)
    yield ("done", None)


ALGORITHMS = {
    "A* Search": astar,
    "Dijkstra": dijkstra,
    "Breadth-First Search": bfs,
    "Depth-First Search": dfs,
}
IGNORES_WEIGHTS = {"Breadth-First Search", "Depth-First Search"}


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
class PathfinderApp:
    def __init__(self, root):
        self.root = root
        root.title("Pathfinding Visualizer")
        root.configure(bg=COLORS["bg"])
        root.resizable(False, False)

        self.start = (ROWS // 2, COLS // 4)
        self.end = (ROWS // 2, 3 * COLS // 4)
        self.walls = set()
        self.weights = {}
        self.marks = {}   # node -> "frontier" | "visit" | "path"
        self.rects = {}   # node -> canvas rectangle id

        self.drag = None      # "start" | "end" | "draw" | None
        self.erasing = False
        self.busy = False
        self.job = None
        self.gen = None
        self.path = None
        self.path_i = 0
        self.visited = 0

        self.algo_var = tk.StringVar(value="A* Search")
        self.tool_var = tk.StringVar(value="wall")
        self.speed_var = tk.DoubleVar(value=70)
        self.status_var = tk.StringVar(value="Draw some walls, then press Visualize.")

        self._build_ui()
        self._draw_grid()

    # ---- layout ------------------------------------------------------------
    def _build_ui(self):
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="Algorithm:").pack(side="left")
        self.algo_box = ttk.Combobox(
            bar, textvariable=self.algo_var, values=list(ALGORITHMS),
            state="readonly", width=20,
        )
        self.algo_box.pack(side="left", padx=(4, 10))

        self.go_btn = ttk.Button(bar, text="Visualize", command=self.visualize)
        self.go_btn.pack(side="left", padx=2)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        self.path_btn = ttk.Button(bar, text="Clear Path", command=self.clear_path)
        self.path_btn.pack(side="left", padx=2)
        self.board_btn = ttk.Button(bar, text="Clear Board", command=self.clear_board)
        self.board_btn.pack(side="left", padx=2)
        self.maze_btn = ttk.Button(bar, text="Random Maze", command=self.random_maze)
        self.maze_btn.pack(side="left", padx=2)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(bar, text="Tool:").pack(side="left")
        for text, value in (("Wall", "wall"), (f"Weight ({WEIGHT_COST})", "weight"), ("Erase", "erase")):
            ttk.Radiobutton(bar, text=text, value=value, variable=self.tool_var).pack(side="left", padx=3)

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)

        ttk.Label(bar, text="Speed:").pack(side="left")
        ttk.Scale(bar, from_=1, to=100, variable=self.speed_var, length=110).pack(side="left", padx=4)

        self.canvas = tk.Canvas(
            self.root, width=COLS * CELL, height=ROWS * CELL,
            bg=COLORS["empty"], highlightthickness=0,
        )
        self.canvas.pack(padx=10)
        self.canvas.bind("<Button-1>", lambda e: self._on_press(e, False))
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        for btn in ("<Button-2>", "<Button-3>"):  # right-click (Button-2 on macOS)
            self.canvas.bind(btn, lambda e: self._on_press(e, True))
        for btn in ("<B2-Motion>", "<B3-Motion>"):
            self.canvas.bind(btn, self._on_motion)
        for btn in ("<ButtonRelease-2>", "<ButtonRelease-3>"):
            self.canvas.bind(btn, self._on_release)

        legend = tk.Frame(self.root, bg=COLORS["bg"])
        legend.pack(fill="x", padx=10, pady=(6, 0))
        for label, key in (
            ("Start", "start"), ("End", "end"), ("Wall", "wall"), ("Weight", "weight"),
            ("Frontier", "frontier"), ("Visited", "visit"), ("Path", "path"),
        ):
            tk.Label(legend, bg=COLORS[key], width=2, relief="solid", bd=1).pack(side="left", padx=(0, 3))
            tk.Label(legend, text=label, bg=COLORS["bg"]).pack(side="left", padx=(0, 12))

        tk.Label(
            self.root, textvariable=self.status_var, anchor="w", bg=COLORS["bg"],
            font=("Segoe UI", 10, "bold"),
        ).pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(
            self.root, anchor="w", bg=COLORS["bg"], fg="#666",
            text="Left-drag: draw   |   Right-drag: erase   |   Drag the green / red cells to move start / end",
        ).pack(fill="x", padx=10, pady=(0, 8))

    def _draw_grid(self):
        for r in range(ROWS):
            for c in range(COLS):
                x, y = c * CELL, r * CELL
                self.rects[(r, c)] = self.canvas.create_rectangle(
                    x, y, x + CELL, y + CELL, fill=COLORS["empty"], outline=COLORS["grid"]
                )
        self._refresh(self.start)
        self._refresh(self.end)

    # ---- drawing helpers ---------------------------------------------------
    def _color(self, node):
        if node == self.start:
            return COLORS["start"]
        if node == self.end:
            return COLORS["end"]
        if node in self.walls:
            return COLORS["wall"]
        mark = self.marks.get(node)
        weighted = node in self.weights
        if mark == "path":
            return COLORS["path"]
        if mark == "visit":
            return COLORS["visit_w"] if weighted else COLORS["visit"]
        if mark == "frontier":
            return COLORS["frontier"]
        if weighted:
            return COLORS["weight"]
        return COLORS["empty"]

    def _refresh(self, node):
        self.canvas.itemconfig(self.rects[node], fill=self._color(node))

    def _set_mark(self, node, mark):
        if mark == "frontier" and self.marks.get(node) == "visit":
            return
        self.marks[node] = mark
        self._refresh(node)

    def _set_busy(self, busy):
        self.busy = busy
        idle = "disabled" if busy else "normal"
        self.go_btn.config(state=idle)
        self.path_btn.config(state=idle)
        self.board_btn.config(state=idle)
        self.maze_btn.config(state=idle)
        self.algo_box.config(state="disabled" if busy else "readonly")
        self.stop_btn.config(state="normal" if busy else "disabled")

    # ---- mouse handling ----------------------------------------------------
    def _cell_at(self, event):
        c, r = event.x // CELL, event.y // CELL
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else None

    def _on_press(self, event, right_click):
        if self.busy:
            return
        cell = self._cell_at(event)
        if cell is None:
            return
        if self.marks:
            self.clear_path()
        self.erasing = right_click
        if cell == self.start and not right_click:
            self.drag = "start"
        elif cell == self.end and not right_click:
            self.drag = "end"
        else:
            self.drag = "draw"
            self._apply_tool(cell)

    def _on_motion(self, event):
        if self.busy or self.drag is None:
            return
        cell = self._cell_at(event)
        if cell is None:
            return
        if self.drag == "draw":
            self._apply_tool(cell)
        elif self.drag == "start":
            self._move_endpoint("start", cell, self.end)
        elif self.drag == "end":
            self._move_endpoint("end", cell, self.start)

    def _on_release(self, _event):
        self.drag = None

    def _move_endpoint(self, which, cell, other):
        if cell == other or cell in self.walls:
            return
        old = getattr(self, which)
        if cell == old:
            return
        setattr(self, which, cell)
        self._refresh(old)
        self._refresh(cell)

    def _apply_tool(self, cell):
        if cell in (self.start, self.end):
            return
        tool = "erase" if self.erasing else self.tool_var.get()
        if tool == "erase":
            self.walls.discard(cell)
            self.weights.pop(cell, None)
        elif tool == "wall":
            self.weights.pop(cell, None)
            self.walls.add(cell)
        else:  # weight
            self.walls.discard(cell)
            self.weights[cell] = WEIGHT_COST
        self._refresh(cell)

    # ---- board actions -----------------------------------------------------
    def clear_path(self):
        if self.busy:
            return
        old = list(self.marks)
        self.marks.clear()
        for node in old:
            self._refresh(node)
        self.status_var.set("Path cleared.")

    def clear_board(self):
        if self.busy:
            return
        touched = set(self.marks) | self.walls | set(self.weights)
        self.marks.clear()
        self.walls.clear()
        self.weights.clear()
        for node in touched:
            self._refresh(node)
        self.status_var.set("Board cleared.")

    def random_maze(self):
        if self.busy:
            return
        self.clear_board()
        for r in range(ROWS):
            for c in range(COLS):
                node = (r, c)
                if node in (self.start, self.end):
                    continue
                if random.random() < 0.28:
                    self.walls.add(node)
                    self._refresh(node)
        self.status_var.set("Random maze generated.")

    # ---- running an algorithm ----------------------------------------------
    def _neighbors(self, node):
        r, c = node
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in self.walls:
                yield (nr, nc)

    def _cost(self, node):
        return self.weights.get(node, 1)

    def _steps_per_tick(self):
        return 1 + int(self.speed_var.get()) // 20

    def _delay(self):
        return max(1, int((100 - self.speed_var.get()) * 0.6))

    def visualize(self):
        if self.busy:
            return
        self.clear_path()
        name = self.algo_var.get()
        self.gen = ALGORITHMS[name](self.start, self.end, self._neighbors, self._cost)
        self.visited = 0
        self._set_busy(True)
        note = " (ignores weights)" if name in IGNORES_WEIGHTS and self.weights else ""
        self.status_var.set(f"Running {name}...{note}")
        self._tick()

    def _tick(self):
        if not self.busy:
            return
        for _ in range(self._steps_per_tick()):
            try:
                kind, value = next(self.gen)
            except StopIteration:
                self._finish(None)
                return
            if kind == "frontier":
                self._set_mark(value, "frontier")
            elif kind == "visit":
                self.visited += 1
                self._set_mark(value, "visit")
            else:  # done
                if value is None:
                    self._finish(None)
                else:
                    self.path, self.path_i = value, 0
                    self.job = self.root.after(self._delay(), self._animate_path)
                return
        self.job = self.root.after(self._delay(), self._tick)

    def _animate_path(self):
        if not self.busy:
            return
        if self.path_i < len(self.path):
            self._set_mark(self.path[self.path_i], "path")
            self.path_i += 1
            self.job = self.root.after(25, self._animate_path)
        else:
            self._finish(self.path)

    def _finish(self, path):
        self._set_busy(False)
        self.job = None
        if path is None:
            self.status_var.set(f"No path found.  Nodes visited: {self.visited}")
        else:
            steps = len(path) - 1
            total = sum(self._cost(n) for n in path[1:])
            self.status_var.set(
                f"Done!  Nodes visited: {self.visited}   |   Path length: {steps} steps   |   Path cost: {total}"
            )

    def stop(self):
        if not self.busy:
            return
        if self.job is not None:
            self.root.after_cancel(self.job)
            self.job = None
        self._set_busy(False)
        self.status_var.set("Stopped.")


def main():
    root = tk.Tk()
    PathfinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()