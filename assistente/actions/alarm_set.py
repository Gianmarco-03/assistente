"""Handler for the ``alarm_set`` intent."""

from __future__ import annotations

from .utils import save_state, load_state

def handle(args: dict) -> str:
    res = (f"sveglia impostata alle ")
    simulation = load_state()
    for time in args['time']:
        if time not in simulation['alarms']:
            simulation['alarms'].append(args['time'])
            res += time + ' '
    save_state(simulation)
    if res == (f"sveglia impostata alle "):
        return "Le sveglie erno giá impostate"
    return res


