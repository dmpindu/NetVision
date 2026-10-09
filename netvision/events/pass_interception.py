"""Turn the possession timeline into game events.

When the holder changes from one player to another:

* same team -> pass
* different team -> interception
* a shot was released in between -> rebound (offensive or defensive)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from ..possession import NO_PLAYER

PASS = "pass"
INTERCEPTION = "interception"
REBOUND = "rebound"
SHOT = "shot"


@dataclass
class Event:
    type: str
    frame: int
    team: int  # team credited with the event (receiver / shooter)
    from_player: Optional[int] = None
    to_player: Optional[int] = None
    from_team: Optional[int] = None
    to_team: Optional[int] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PassInterceptionDetector:
    def detect(
        self,
        possession: Sequence[int],
        team_assignment: dict[int, int],
        shot_release_frames: Optional[dict[int, int]] = None,
    ) -> list[Event]:
        """Return pass, interception, and rebound events in frame order.

        ``shot_release_frames`` maps release frame -> shooter ID. A holder
        change that spans a shot release by the previous holder is a rebound.
        """
        shot_release_frames = shot_release_frames or {}
        events: list[Event] = []
        last_holder = NO_PLAYER
        last_holder_frame = -1
        run_start = -1  # first frame of the last holder's current possession run
        for frame, holder in enumerate(possession):
            if holder == NO_PLAYER:
                continue
            if holder != last_holder or last_holder_frame < frame - 1:
                new_run_start = frame
            else:
                new_run_start = run_start
            if last_holder != NO_PLAYER and holder != last_holder:
                from_team = team_assignment.get(last_holder, 0)
                to_team = team_assignment.get(holder, 0)
                # The shot analyzer may place the release a few frames before
                # possession officially ended, so look back to the run start.
                shot_between = any(
                    run_start <= release <= frame and shooter == last_holder
                    for release, shooter in shot_release_frames.items()
                )
                if from_team in (1, 2) and to_team in (1, 2):
                    if shot_between:
                        kind = "offensive" if from_team == to_team else "defensive"
                        events.append(Event(REBOUND, frame, to_team, last_holder, holder, from_team, to_team, kind))
                    elif from_team == to_team:
                        events.append(Event(PASS, frame, to_team, last_holder, holder, from_team, to_team))
                    else:
                        events.append(Event(INTERCEPTION, frame, to_team, last_holder, holder, from_team, to_team))
            elif last_holder == holder and last_holder_frame < frame - 1:
                # Same player got the ball back. After their own shot that is an offensive rebound.
                shot_between = any(
                    run_start <= release <= frame and shooter == holder
                    for release, shooter in shot_release_frames.items()
                )
                team = team_assignment.get(holder, 0)
                if shot_between and team in (1, 2):
                    events.append(Event(REBOUND, frame, team, holder, holder, team, team, "offensive"))
            last_holder = holder
            last_holder_frame = frame
            run_start = new_run_start
        return events


def cumulative_event_counts(events: Sequence[Event], num_frames: int, kinds: Sequence[str] = (PASS, INTERCEPTION)) -> list[dict]:
    """Running count per frame of each event kind per team, used by the drawer."""
    counts = {kind: {1: 0, 2: 0} for kind in kinds}
    by_frame: dict[int, list[Event]] = {}
    for event in events:
        by_frame.setdefault(event.frame, []).append(event)
    output = []
    for frame in range(num_frames):
        for event in by_frame.get(frame, []):
            if event.type in counts and event.team in (1, 2):
                counts[event.type][event.team] += 1
        output.append({kind: dict(values) for kind, values in counts.items()})
    return output
