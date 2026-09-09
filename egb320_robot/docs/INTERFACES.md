# Interface contracts (the arrows on the architecture diagram)

Five lines per arrow: direction · payload + units + frame · rate/trigger · failure behaviour · status.
Code: `egb320/interfaces/`. Status tags: PLN planned · PRO prototyped · IMP implemented · TST tested.

| # | Arrow | Payload (units, frame) | Rate / trigger | If absent / stale / wrong | Status |
|---|---|---|---|---|---|
| 1 | Vision → Nav | `DetectionFrame`: list of (class, range m, bearing rad +left, confidence) | streamed ≥ 10 Hz | empty list is normal; stale (>0.5 s) treated as empty; act only after N consecutive frames | PLN |
| 2 | Nav → Mobility | `VelocityCommand` (v m/s, ω rad/s +left) | 20 Hz while driving | mobility clamps; watchdog stops motors if no command for 0.5 s (**agree with Dan**) | PLN |
| 3 | Mobility → Nav | `EncoderSample` (signed cumulative ticks L/R) or `Pose2D` | 20 Hz | nav falls back to commanded-motion dead reckoning + stall watchdog → RECOVER | PLN |
| 4 | Ultrasonics → Nav | `WallRanges` (m, left/front/right, None = nothing) | ≤ 20 Hz sequential | None readings = no wall; collision_stop distance always honoured | PLN |
| 5 | Nav → Rescue | `collect()` / `release()` / `clear_rubble()` | on state entry | — | PLN |
| 6 | Rescue → Nav | `RescueStatus` (BUSY/DONE/FAILED, has_victim) | polled 20 Hz | timeout `rescue_timeout_s` → treat as FAILED, retry then give up | PLN |
| 7 | Nav → LEDs | `LedState` Y/G/R | on every state change | — | PLN |
| 8 | ? → Nav | GO signal | once | **TODO decide**: button / switch / first frame | PLN |

Open decisions (from the 2026-08-18 log): who integrates ticks → pose; ToF vs camera for wall sensing;
bearing sign convention; encoder decode ×2/×4; Camera Module 3 FOV variant.
