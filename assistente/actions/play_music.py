"""Handler for the ``play_music`` intent."""

from __future__ import annotations

from .utils import load_state, save_state

def handle(args: dict) -> str:
    if args == None:
        return "cosa vuoi sentire?"
    simulation = load_state()
    music = simulation["music"]
    known_songs = simulation['known_songs']
    res = ""
    try:
        song_name = args['music_name'][0].lower()
    except:
        try:
            genre = args['music_genre'][0].lower()
        except:
            try:
                author = args['artist_name'][0].lower()
            except:
                try:
                    playlist = args['playlist_name'].lower()
                except:
                    return "scusa, non ho capito"
                else:
                    for song in known_songs:
                        if known_songs[song]['playlist'] == playlist:
                            music = known_songs[song]
                            music['song'] = song  
                            res = (f"riproduco {music['song']}")
                            break
            else:
                for song in known_songs:
                    if known_songs[song]['autore'] == author:
                        music = known_songs[song]
                        music['song'] = song
                        res = (f"riproduco {music['song']}")
                        break
        else:
            for song in known_songs:
                if known_songs[song]['genre'] == genre:
                    music = known_songs[song]
                    music['song'] = song
                    res = (f"riproduco {music['song']}")
                    break
    else:
        if song_name in known_songs:
            music = known_songs[args['music_name']]
            music['song'] = args['music_name']
            res = (f"riproduco {music['song']}")
    simulation['music'] = music
    save_state(simulation)
    return res