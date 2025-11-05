"""Handler for the ``iot_hue_lightup`` intent."""

from __future__ import annotations
from actions import iot_hue_lighton as on
from .utils import save_state, load_state


def handle(args: dict) -> str:
    simulation = load_state()
    res = ""
    found = []
    not_found = []
    for room in args['house_place']:
        try:
            light = simulation['place'][room]
            found.append(room + ', ')
            simulation['place'][room]['isOn'] = True
            simulation['place'][room]['intensity'] += 1
        except :
            not_found.append[room]
    save_state(simulation)
    if len(found) != 0:
        res += "ho acceso la luce in "
        for room in found:
            res += room + ", "
    if len(not_found) != 0:
        res += "non ho trovato le seguenti stanze "
        for room in found:
            res += room + ", "
    return res
