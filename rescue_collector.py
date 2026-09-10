"""
rescue_collector.py — the Navigation-facing interface to the rescue collector.

This is the single boundary between Navigation and Rescue Collection. Nothing
in Navigation should import rescue_arm or rescue_spool directly; it talks to
RescueCollector and nothing else. On the architecture diagram this module IS
the "collection command" interface, and the status properties below are the
"capture-confirmed" and "deposit-confirmed" returns [M2].

Why the call does not block
---------------------------
A capture cycle takes several seconds. If Navigation blocked for that time:
  - the vision loop would stop
  - LED state could not be updated
  - a jammed mechanism would hang the whole run, with the 7 minute clock
    still running and no way to abort

So start_collect() returns immediately and the sequence runs on a worker
thread. Navigation polls state each loop. A blocking helper is provided for
bench testing only.

Sequencing
----------
The arm and spool act in strict sequence, never together (DEC-06). Cinching
mid-swing closes the bag before it reaches the victim. The state machine
enforces this ordering.

    IDLE -> DEPLOYING -> CINCHING -> STOWING -> HOLDING
                                                  |
                                        start_release()
                                                  v
                                             RELEASING -> IDLE

LED ownership
-------------
This module does NOT drive the LEDs. It exposes led_hint() so whichever
subsystem owns the LED outputs can read the collector's contribution. Green
must be an autonomous response to detection and must never be pre-lit or left
continuous [D3.1] -- detection lives in Vision, collection state lives here,
and the LED is a system output. Agree ownership across the three before M3;
it is a likely interview target.

Known gap: capture confirmation
-------------------------------
A continuous-rotation spool servo reports nothing about its own position, so
cinching is timed and open-loop. capture_confirmed is therefore ASSUMED, not
sensed, unless a confirm hook is attached. Set collector.confirm_hook to a
callable returning True once a microswitch, current sense or encoder detects
closure. Until then, treat the flag as "sequence completed", not "victim
captured", and say so when asked.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable, Optional


class RescueState(Enum):
    IDLE = "idle"                # nothing in progress, no payload
    DEPLOYING = "deploying"      # arm swinging down to the victim
    CINCHING = "cinching"        # drawstring closing
    STOWING = "stowing"          # arm lifting with payload
    HOLDING = "holding"          # payload retained, safe to drive
    RELEASING = "releasing"      # paying out over the base zone
    FAILED = "failed"            # timeout or abort; needs a reset


class RescueCollector:
    """Facade over the arm and spool.

    Typical use from Navigation:

        collector = RescueCollector(arm_gpio=13, spool_gpio=12)
        collector.park()                       # once, at startup

        # ... on reaching a victim ...
        collector.start_collect()

        while True:                            # Navigation's own loop
            vision.update()
            if collector.state is RescueState.HOLDING:
                break
            if collector.state is RescueState.FAILED:
                handle_failure()
                break
            time.sleep(0.02)
    """

    def __init__(self, arm_gpio: int = 13, spool_gpio: int = 12,
                 timeout_s: float = 20.0, simulate: bool = False):
        self.timeout_s = timeout_s
        self.simulate = simulate

        self._state = RescueState.IDLE
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._abort = threading.Event()
        self._error: Optional[str] = None

        # Attach a callable returning True once capture is physically sensed.
        # Until one exists, capture_confirmed reflects sequence completion.
        self.confirm_hook: Optional[Callable[[], bool]] = None

        self._arm = None
        self._spool = None
        if not simulate:
            from rescue_arm import Arm, load_cal as load_arm_cal
            from rescue_spool import Spool, load_cal as load_spool_cal
            self._arm = Arm(load_arm_cal(arm_gpio))
            self._spool = Spool(load_spool_cal(spool_gpio))
            self._spool.confirm_hook = lambda: (
                self.confirm_hook() if self.confirm_hook else False
            )

    # ---- state, read freely from Navigation -----------------------------

    @property
    def state(self) -> RescueState:
        with self._lock:
            return self._state

    def _set(self, s: RescueState) -> None:
        with self._lock:
            self._state = s

    def is_busy(self) -> bool:
        return self.state in (RescueState.DEPLOYING, RescueState.CINCHING,
                              RescueState.STOWING, RescueState.RELEASING)

    @property
    def has_payload(self) -> bool:
        return self.state is RescueState.HOLDING

    @property
    def capture_confirmed(self) -> bool:
        """True once a capture cycle has completed.

        SENSED if confirm_hook is attached, otherwise ASSUMED from the timed
        sequence completing. Do not present this as sensed confirmation until
        a hook exists.
        """
        return self.state is RescueState.HOLDING

    @property
    def error(self) -> Optional[str]:
        return self._error

    def led_hint(self) -> Optional[str]:
        """This subsystem's contribution to LED state. Not authoritative."""
        s = self.state
        if s in (RescueState.DEPLOYING, RescueState.CINCHING):
            return "green"       # collection in progress [D5.6]
        if s in (RescueState.STOWING, RescueState.HOLDING):
            return "red"         # returning with a payload
        return None              # Navigation decides (yellow while exploring)

    # ---- commands from Navigation ---------------------------------------

    def park(self) -> None:
        """Attach the arm at stow. Call once at startup, arm placed by hand."""
        if self._arm:
            self._arm.park()
        self._set(RescueState.IDLE)

    def start_collect(self) -> bool:
        """Begin a capture. Returns False if busy or already holding."""
        if self.is_busy() or self.state is RescueState.HOLDING:
            return False
        self._error = None
        self._abort.clear()
        self._thread = threading.Thread(target=self._run_collect, daemon=True)
        self._thread.start()
        return True

    def start_release(self) -> bool:
        """Deposit in the base zone. Returns False if there is no payload.

        Release is optional for scoring -- RESCUED counts if the victim is
        merely retained while the robot is inside the zone [D5.4] -- so a
        failure here is not fatal to the mark.
        """
        if self.state is not RescueState.HOLDING:
            return False
        self._error = None
        self._abort.clear()
        self._thread = threading.Thread(target=self._run_release, daemon=True)
        self._thread.start()
        return True

    def abort(self) -> None:
        """Request a stop. The current motion finishes, then it halts."""
        self._abort.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the current operation ends. Bench testing only."""
        if self._thread:
            self._thread.join(timeout or self.timeout_s)
        return not self.is_busy()

    def collect_blocking(self, timeout: Optional[float] = None) -> bool:
        """Convenience wrapper. Do not use inside Navigation's control loop."""
        if not self.start_collect():
            return False
        self.wait(timeout)
        return self.state is RescueState.HOLDING

    def close(self) -> None:
        self.abort()
        if self._thread:
            self._thread.join(2.0)
        if self._arm:
            self._arm.close()
        if self._spool:
            self._spool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- worker ----------------------------------------------------------

    def _fail(self, msg: str) -> None:
        self._error = msg
        self._set(RescueState.FAILED)

    def _run_collect(self) -> None:
        t0 = time.monotonic()
        try:
            self._set(RescueState.DEPLOYING)
            if self.simulate:
                time.sleep(1.0)
            else:
                self._arm.deploy()
            if self._abort.is_set():
                return self._fail("aborted during deploy")
            if time.monotonic() - t0 > self.timeout_s:
                return self._fail("timeout during deploy")

            self._set(RescueState.CINCHING)
            if self.simulate:
                time.sleep(1.5)
            else:
                self._spool.cinch()
            if self._abort.is_set():
                return self._fail("aborted during cinch")

            self._set(RescueState.STOWING)
            if self.simulate:
                time.sleep(1.0)
            else:
                self._arm.stow()

            if time.monotonic() - t0 > self.timeout_s:
                return self._fail("timeout; payload state unknown")
            self._set(RescueState.HOLDING)

        except Exception as exc:                      # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")

    def _run_release(self) -> None:
        try:
            self._set(RescueState.RELEASING)
            if self.simulate:
                time.sleep(1.2)
            else:
                self._spool.release()
            self._set(RescueState.IDLE)
        except Exception as exc:                      # noqa: BLE001
            self._fail(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sim = "--sim" in sys.argv
    print(f"collector self-test{' (simulated)' if sim else ''}\n")
    with RescueCollector(simulate=sim) as c:
        if not sim:
            print("place the arm near stow, then Enter")
            input()
        c.park()
        c.start_collect()
        last = None
        while c.is_busy():
            if c.state is not last:
                last = c.state
                print(f"  {last.value:<10} led={c.led_hint() or '-'}")
            time.sleep(0.05)
        print(f"  {c.state.value:<10} led={c.led_hint() or '-'}")
        if c.error:
            print(f"\nerror: {c.error}")
        elif c.has_payload:
            print("\npayload held. Enter to release")
            input()
            c.start_release()
            c.wait()
            print(f"  {c.state.value}")
