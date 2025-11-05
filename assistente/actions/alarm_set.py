"""Handler for the ``alarm_set`` intent."""

from __future__ import annotations

from .utils import save_state, load_state

def handle(args: dict) -> str:
    simulation = load_state()
    if args['time'] not in simulation['alarms']:
        simulation['alarms'].append(args['time'])
    save_state(simulation)

    return (f"sveglia impostata alle {args['time']}")

