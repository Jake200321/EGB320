"""Draws what nav knows about the maze: the second window beside the camera.

    walls seen          white
    not seen yet        faint grey grid
    visited cells       blue
    frontier            yellow outline -- where exploring will go next
    planned route       cyan line
    base                green square
    victims picked up   magenta dot
    robot               orange arrow at its odometry pose
    blocked cells       red cross

Separate from the window itself so it can be rendered and saved to a PNG without a
display -- main.py writes one at the end of every run.
"""

import math

PX_PER_CELL = 64
MARGIN = 20
PANEL_H = 70

BG = (30, 30, 30)
GRID = (70, 70, 70)
WALL = (255, 255, 255)
VISITED = (110, 60, 20)
FRONTIER = (0, 220, 255)
ROUTE = (255, 255, 0)
BASE = (0, 170, 0)
VICTIM = (255, 0, 255)
ROBOT = (0, 140, 255)
BLOCKED = (0, 0, 255)
TEXT = (230, 230, 230)


def render_map(nav):
    """One BGR image of the map, the robot and the plan."""
    import cv2
    import numpy as np

    m = nav.map
    w = m.cols * PX_PER_CELL + 2 * MARGIN
    h = m.rows * PX_PER_CELL + 2 * MARGIN + PANEL_H
    img = np.full((h, w, 3), BG, np.uint8)

    def corner(col, row):
        return (MARGIN + col * PX_PER_CELL, MARGIN + row * PX_PER_CELL)

    def cell_rect(cell, inset=0):
        x0, y0 = corner(*cell)
        return ((x0 + inset, y0 + inset),
                (x0 + PX_PER_CELL - inset, y0 + PX_PER_CELL - inset))

    def world_px(x, y):
        return (int(MARGIN + x / m.cell_m * PX_PER_CELL),
                int(MARGIN + (m.rows - y / m.cell_m) * PX_PER_CELL))

    def cell_mid(cell):
        (x0, y0), (x1, y1) = cell_rect(cell)
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    for cell in m.visited:
        cv2.rectangle(img, *cell_rect(cell), VISITED, -1)
    cv2.rectangle(img, *cell_rect(nav.base_cell, 6), BASE, -1)
    for cell in m.frontier_cells():
        cv2.rectangle(img, *cell_rect(cell, 4), FRONTIER, 1)
    for cell in m.blocked:
        (x0, y0), (x1, y1) = cell_rect(cell, 10)
        cv2.line(img, (x0, y0), (x1, y1), BLOCKED, 2)
        cv2.line(img, (x0, y1), (x1, y0), BLOCKED, 2)

    for col in range(m.cols + 1):
        cv2.line(img, corner(col, 0), corner(col, m.rows), GRID, 1)
    for row in range(m.rows + 1):
        cv2.line(img, corner(0, row), corner(m.cols, row), GRID, 1)

    # A wall on side `heading` of a cell runs between two of its corners.
    ends = {0: ((0, 0), (1, 0)), 1: ((1, 0), (1, 1)),
            2: ((0, 1), (1, 1)), 3: ((0, 0), (0, 1))}
    for cell, heading in m.known_walls():
        (a, b) = ends[heading]
        cv2.line(img, corner(cell[0] + a[0], cell[1] + a[1]),
                 corner(cell[0] + b[0], cell[1] + b[1]), WALL, 4)

    route = getattr(nav, "route", None)
    if route and len(route) > 1:
        pts = [cell_mid(c) for c in route]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, ROUTE, 2)

    for x, y in getattr(nav, "victim_points", []):
        cv2.circle(img, world_px(x, y), 7, VICTIM, -1)

    # --- the robot, at its odometry pose ---
    odo = nav.odo
    cx, cy = world_px(odo.x, odo.y)
    r = PX_PER_CELL * 0.28
    th = odo.theta
    tip = (int(cx + r * math.cos(th)), int(cy - r * math.sin(th)))
    back_l = (int(cx + r * 0.8 * math.cos(th + 2.5)), int(cy - r * 0.8 * math.sin(th + 2.5)))
    back_r = (int(cx + r * 0.8 * math.cos(th - 2.5)), int(cy - r * 0.8 * math.sin(th - 2.5)))
    cv2.fillPoly(img, [np.array([tip, back_l, back_r], np.int32)], ROBOT)

    # --- status panel ---
    y0 = h - PANEL_H + 18
    label = getattr(nav, "state_label", lambda: nav.state)()
    cv2.putText(img, label, (MARGIN, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1)
    explored = f"explored {len(m.visited)}/{m.cols * m.rows}"
    rescued = f"rescued {getattr(nav, 'rescued', 0)}"
    cell = getattr(nav, "cell", None)
    where = "" if cell is None else f"cell {cell}"
    cv2.putText(img, f"{explored}   {rescued}   {where}", (MARGIN, y0 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
    elapsed = getattr(nav, "elapsed", lambda: None)()
    if elapsed is not None:
        cv2.putText(img, f"{int(elapsed // 60)}:{int(elapsed % 60):02d}",
                    (w - MARGIN - 50, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT, 1)
    cv2.putText(img, f"L {odo.left_m:5.2f}m  R {odo.right_m:5.2f}m",
                (MARGIN, y0 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT, 1)
    return img
