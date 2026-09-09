"""The state machine has no hardware, so it is the easiest thing on the robot to test."""

from egb320.interfaces.messages import RescueOutcome, RescueStatus, WallRanges
from egb320.navigation.state_machine import NavState, NavStateMachine, Observation


def obs(t=0.0, **kw):
    base = dict(t=t, detections=None, walls=WallRanges(t, None, None, None),
                rescue=RescueStatus(RescueOutcome.IDLE, False))
    base.update(kw)
    return Observation(**base)


def test_boot_then_wait_then_go():
    fsm = NavStateMachine(time_low_s=390)
    assert fsm.state is NavState.BOOT
    fsm.step(obs())
    assert fsm.state is NavState.WAIT_FOR_GO
    fsm.step(obs(go_signal=True))
    assert fsm.state is NavState.EXPLORE
    assert len(fsm.history) == 2


# TODO as you add arrows: victim seen -> APPROACH, range <= 10 cm -> COLLECT,
# DONE+has_victim -> RETURN, time-low while carrying -> RETURN, etc.
