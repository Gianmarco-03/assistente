"""Azioni associate alle classi di intent riconosciute dal modello."""

from __future__ import annotations

from typing import Callable, Dict


def _static(message: str) -> Callable[[], str]:
    def _action() -> str:
        return message

    return _action


_INTENT_ACTIONS: Dict[str, Callable[[], str]] = {
    "alarm_query": _static("Questa richiesta chiede informazioni sulle sveglie impostate."),
    "alarm_remove": _static("Questa richiesta vuole cancellare una sveglia."),
    "alarm_set": _static("Questa richiesta desidera impostare una nuova sveglia."),
    "audio_volume_down": _static("Questa richiesta chiede di abbassare il volume."),
    "audio_volume_mute": _static("Questa richiesta chiede di silenziare l'audio."),
    "audio_volume_other": _static("Questa richiesta riguarda un'impostazione particolare del volume."),
    "audio_volume_up": _static("Questa richiesta chiede di alzare il volume."),
    "calendar_query": _static("Questa richiesta chiede informazioni sul calendario."),
    "calendar_remove": _static("Questa richiesta vuole rimuovere un evento dal calendario."),
    "calendar_set": _static("Questa richiesta vuole aggiungere o programmare un evento nel calendario."),
    "cooking_query": _static("Questa richiesta riguarda domande generiche sulla cucina."),
    "cooking_recipe": _static("Questa richiesta chiede una ricetta di cucina."),
    "datetime_convert": _static("Questa richiesta vuole convertire una data o un orario."),
    "datetime_query": _static("Questa richiesta chiede l'ora o la data."),
    "email_addcontact": _static("Questa richiesta chiede di aggiungere un contatto e-mail."),
    "email_query": _static("Questa richiesta chiede informazioni sulle e-mail."),
    "email_querycontact": _static("Questa richiesta chiede informazioni sui contatti e-mail."),
    "email_sendemail": _static("Questa richiesta vuole inviare un messaggio e-mail."),
    "general_greet": _static("Questa richiesta è un saluto."),
    "general_joke": _static("Questa richiesta chiede di raccontare una barzelletta."),
    "general_quirky": _static("Questa richiesta è una domanda curiosa o stravagante."),
    "iot_cleaning": _static("Questa richiesta controlla un dispositivo di pulizia domestica."),
    "iot_coffee": _static("Questa richiesta vuole preparare o gestire la macchina del caffè."),
    "iot_hue_lightchange": _static("Questa richiesta chiede di modificare il colore delle luci intelligenti."),
    "iot_hue_lightdim": _static("Questa richiesta chiede di abbassare l'intensità delle luci intelligenti."),
    "iot_hue_lightoff": _static("Questa richiesta chiede di spegnere le luci intelligenti."),
    "iot_hue_lighton": _static("Questa richiesta chiede di accendere le luci intelligenti."),
    "iot_hue_lightup": _static("Questa richiesta chiede di aumentare la luminosità delle luci intelligenti."),
    "iot_wemo_off": _static("Questa richiesta chiede di spegnere un dispositivo collegato a una presa intelligente."),
    "iot_wemo_on": _static("Questa richiesta chiede di accendere un dispositivo collegato a una presa intelligente."),
    "lists_createoradd": _static("Questa richiesta chiede di creare o aggiungere elementi a una lista."),
    "lists_query": _static("Questa richiesta chiede informazioni su una lista."),
    "lists_remove": _static("Questa richiesta chiede di rimuovere elementi da una lista."),
    "music_dislikeness": _static("Questa richiesta esprime disapprezzamento per un brano musicale."),
    "music_likeness": _static("Questa richiesta esprime apprezzamento per un brano musicale."),
    "music_query": _static("Questa richiesta chiede informazioni su musica o brani."),
    "music_settings": _static("Questa richiesta chiede di modificare impostazioni legate alla musica."),
    "news_query": _static("Questa richiesta chiede notizie o aggiornamenti."),
    "play_audiobook": _static("Questa richiesta chiede di riprodurre un audiolibro."),
    "play_game": _static("Questa richiesta chiede di avviare un gioco."),
    "play_music": _static("Questa richiesta chiede di riprodurre musica."),
    "play_podcasts": _static("Questa richiesta chiede di riprodurre un podcast."),
    "play_radio": _static("Questa richiesta chiede di riprodurre la radio."),
    "qa_currency": _static("Questa richiesta chiede informazioni su valute o conversioni monetarie."),
    "qa_definition": _static("Questa richiesta chiede la definizione di un termine."),
    "qa_factoid": _static("Questa richiesta chiede un fatto o una curiosità."),
    "qa_maths": _static("Questa richiesta chiede di risolvere un problema matematico."),
    "qa_stock": _static("Questa richiesta chiede informazioni su titoli azionari."),
    "recommendation_events": _static("Questa richiesta chiede suggerimenti su eventi a cui partecipare."),
    "recommendation_locations": _static("Questa richiesta chiede suggerimenti su luoghi da visitare."),
    "recommendation_movies": _static("Questa richiesta chiede suggerimenti su film da guardare."),
    "social_post": _static("Questa richiesta chiede di pubblicare un aggiornamento sui social."),
    "social_query": _static("Questa richiesta chiede informazioni dai social network."),
    "takeaway_order": _static("Questa richiesta chiede di ordinare cibo da asporto."),
    "takeaway_query": _static("Questa richiesta chiede informazioni su ordini di cibo da asporto."),
    "transport_query": _static("Questa richiesta chiede informazioni su percorsi o orari di trasporto."),
    "transport_taxi": _static("Questa richiesta chiede di prenotare o chiamare un taxi."),
    "transport_ticket": _static("Questa richiesta chiede di acquistare o gestire un biglietto di viaggio."),
    "transport_traffic": _static("Questa richiesta chiede informazioni sul traffico."),
    "weather_query": _static("Questa richiesta chiede informazioni sulle condizioni meteo."),
}


def handle(intent: str) -> str:
    """Restituisce il messaggio associato all'intent.

    Se l'intent non è riconosciuto viene restituita una frase generica.
    """

    action = _INTENT_ACTIONS.get(intent)
    if action is not None:
        return action()
    normalized = intent.replace("_", " ")
    return f"Questa richiesta appartiene alla classe '{normalized}'."


__all__ = ["handle"]
