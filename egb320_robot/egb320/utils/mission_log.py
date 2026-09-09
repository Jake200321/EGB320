"""Black-box recorder: every state transition, LED change, pose sample and event,
written to logs/<name>_<timestamp>.json at the end of a run. On demo day this is
your evidence; for the report it's your plots."""

import json
import os
import time


class MissionLog:
    def __init__(self, name: str, out_dir: str = "logs"):
        self.name = name
        self.out_dir = out_dir
        self.transitions = []   # (t, state, why)
        self.events = []        # (t, kind, detail)
        self.trajectory = []    # (t, x, y, heading)

    def transition(self, t, state, why):
        self.transitions.append((round(t, 2), state.name, why))
        print(f"[{t:6.1f}s] -> {state.name}  ({why})")

    def event(self, t, kind, detail=""):
        self.events.append((round(t, 2), kind, detail))

    def pose(self, t, pose):
        self.trajectory.append((round(t, 2), pose.x, pose.y, pose.heading))

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.name}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w") as f:
            json.dump({"transitions": self.transitions, "events": self.events,
                       "trajectory": self.trajectory}, f)
        print(f"Mission log written to {path}")
        return path
