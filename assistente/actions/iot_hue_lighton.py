"""Handler for the ``iot_hue_lighton`` intent."""

from __future__ import annotations
from .utils import save_state, load_state


def handle(args : dict) -> str:
    simulation = load_state()
    if args['house_place'] in simulation['place']:
        simulation['alarms'].append(args['time'])
    else: 
        return (f'la stanza {args['location']} non esiste (idiota).')

    save_state(simulation)
    return (f'luce accesa in {args['location']}.')
