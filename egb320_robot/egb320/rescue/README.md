# Rescue collection (Roger) — what navigation needs from this folder

Nav stops the robot within 10 cm of the victim and then calls:

| Method | Contract |
|---|---|
| `collect()` | start the capture routine, return immediately (non-blocking) |
| `release()` | deposit the victim in the base zone, non-blocking |
| `clear_rubble()` | Level 3: move the rubble; may be a no-op until the mechanism exists |
| `status()` | `RescueStatus(outcome, has_victim)` — outcome is BUSY while running, then DONE or FAILED |
| `abort()` | stop and go to a safe pose |

Nav polls `status()` every tick. It has a fallback timeout (`NavConfig.rescue_timeout_s`) but
**your DONE/FAILED is the real signal** — without it nav either drives home empty or never leaves.
Victims must be lifted/captured/contained, never pushed or dragged (rules).
`stub.py` reports DONE + has_victim a second after `collect()` so the stack runs before your code lands.
