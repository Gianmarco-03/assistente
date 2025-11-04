"""Interfaccia a riga di comando per il progetto Assistente."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from app import create_analyzer_for_file, create_qt_application, playback_audio_file, run_visualizer
from audio.analyzer import AudioAnalyzer
from audio.config import AudioConfig
from audio.microphone import InteractiveAudioSource, MicrophoneSource
from audio.recognizer import BackgroundRecognizer, RecognitionConfig
from intent_router import IntentRouter, IntentRouterError, get_annot_utt
from tts.pyttsx3_engine import Pyttsx3Engine
from tts.responder import TextToSpeechResponder
from visualization.sphere import SphereVisualizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualizzazione e TTS per la sfera audio.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="Riproduce un file audio nella sfera.")
    play_parser.add_argument("audio_path", help="Percorso del file audio (wav, flac, ogg, ...).")
    play_parser.add_argument(
        "--blocksize",
        type=int,
        default=1_024,
        help="Numero di campioni per blocco di analisi (default: 1024).",
    )
    play_parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disabilita il loop della traccia audio una volta terminata.",
    )
    play_parser.set_defaults(func=run_play_command)

    speak_parser = subparsers.add_parser(
        "speak", help="Genera una risposta vocale e anima la sfera con il relativo audio."
    )
    speak_parser.add_argument("text", help="Testo di input per il generatore di risposte.")
    speak_parser.add_argument(
        "--responses-json",
        type=Path,
        help="File JSON contenente mappature personalizzate 'input' -> 'response'.",
    )
    speak_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tts_output"),
        help="Cartella dove salvare i file audio generati (default: tts_output).",
    )
    speak_parser.add_argument(
        "--blocksize",
        type=int,
        default=1_024,
        help="Numero di campioni per blocco di analisi (default: 1024).",
    )
    speak_parser.add_argument(
        "--voice",
        help="ID della voce pyttsx3 da utilizzare (opzionale).",
    )
    speak_parser.add_argument(
        "--rate",
        type=int,
        help="Velocità di parlato per pyttsx3 (opzionale).",
    )
    speak_parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Non riprodurre il file audio generato (rimane disponibile per la sfera).",
    )
    speak_parser.add_argument(
        "--model-path",
        type=Path,
        help="Percorso del modello di classificazione intents (joblib).",
    )
    speak_parser.set_defaults(func=run_speak_command)

    listen_parser = subparsers.add_parser(
        "listen",
        help="Ascolta dal microfono e risponde con voce sintetica mantenendo la sfera attiva.",
    )
    listen_parser.add_argument(
        "--responses-json",
        type=Path,
        help="File JSON con mappature personalizzate 'input' -> 'response'.",
    )
    listen_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tts_output"),
        help="Cartella dove salvare i file audio generati (default: tts_output).",
    )
    listen_parser.add_argument(
        "--blocksize",
        type=int,
        default=1_024,
        help="Numero di campioni per blocco di analisi (default: 1024).",
    )
    listen_parser.add_argument(
        "--voice",
        help="ID della voce pyttsx3 da utilizzare (opzionale).",
    )
    listen_parser.add_argument(
        "--rate",
        type=int,
        help="Velocità di parlato per pyttsx3 (opzionale).",
    )
    listen_parser.add_argument(
        "--samplerate",
        type=float,
        help="Frequenza di campionamento del microfono (default: valore del dispositivo).",
    )
    listen_parser.add_argument(
        "--device",
        help="Nome o indice del dispositivo microfono da utilizzare (opzionale).",
    )
    listen_parser.add_argument(
        "--language",
        default="it-IT",
        help="Codice lingua per il riconoscimento vocale (default: it-IT).",
    )
    listen_parser.add_argument(
        "--start-threshold",
        type=float,
        default=0.02,
        help="Ampiezza media necessaria per iniziare a registrare una frase (default: 0.02).",
    )
    listen_parser.add_argument(
        "--stop-threshold",
        type=float,
        default=0.01,
        help="Ampiezza media sotto cui viene conteggiato il silenzio (default: 0.01).",
    )
    listen_parser.add_argument(
        "--silence-duration",
        type=float,
        default=0.6,
        help="Durata (in secondi) di silenzio necessaria per chiudere una frase (default: 0.6).",
    )
    listen_parser.add_argument(
        "--min-phrase-duration",
        type=float,
        default=0.35,
        help="Durata minima (in secondi) perché una frase venga inviata al riconoscimento (default: 0.35).",
    )
    listen_parser.add_argument(
        "--phrase-time-limit",
        type=float,
        default=6.0,
        help="Durata massima (in secondi) di una frase prima dell'invio forzato (default: 6.0).",
    )
    listen_parser.add_argument(
        "--window-size",
        type=int,
        default=900,
        help="Dimensione della finestra della sfera (default: 900).",
    )
    listen_parser.add_argument(
        "--model-path",
        type=Path,
        help="Percorso del modello di classificazione intents (joblib).",
    )
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Anima la sfera monitorando l'audio di uscita del sistema in tempo reale.",
    )
    monitor_parser.add_argument(
        "--blocksize",
        type=int,
        default=1_024,
        help="Numero di campioni per blocco di analisi (default: 1024).",
    )
    monitor_parser.add_argument(
        "--device",
        help=(
            "Nome o indice del dispositivo di uscita da monitorare (opzionale)."
            " Usa l'elenco di sounddevice per identificarlo."
        ),
    )
    monitor_parser.set_defaults(func=run_monitor_command)

    listen_parser.set_defaults(func=run_listen_command)


    return parser


def run_play_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    config = AudioConfig(blocksize=args.blocksize, loop_track=not args.no_loop)
    analyzer = create_analyzer_for_file(args.audio_path, config)
    return run_visualizer(analyzer)


def load_responses(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("Il file JSON deve contenere un dizionario 'input' -> 'response'.")
    return {str(key): str(value) for key, value in data.items()}

def _summarize_slots(slots: dict[str, list[str]] | None) -> str:
    if not slots:
        return "(nessuno)"
    parts: list[str] = []
    for name in sorted(slots):
        values = [value.strip() for value in slots[name] if value and value.strip()]
        if not values:
            continue
        joined = "; ".join(values)
        parts.append(f"{name}={joined}")
    if not parts:
        return "(nessuno)"
    return ", ".join(parts)



def run_speak_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    responses: dict[str, str] = {}
    if args.responses_json:
        responses = load_responses(args.responses_json)

    config = AudioConfig(blocksize=args.blocksize, loop_track=False)

    engine = Pyttsx3Engine(voice=args.voice, rate=args.rate)
    responder = TextToSpeechResponder(
        engine=engine,
        responses=responses,
        output_dir=args.output_dir,
    )

    response_text, matched = responder.find_response(args.text)
    intent_label: str | None = None
    slot_annotation: str | None = None
    slot_values: dict[str, list[str]] | None = None

    if not matched:
        router = IntentRouter(model_path=args.model_path)
        try:
            intent_label, response_text = router.describe(args.text)
        except IntentRouterError as exc:
            print(f"[errore] Impossibile analizzare la richiesta: {exc}")
        else:
            try:
                slot_annotation, slot_values = get_annot_utt(args.text)
            except IntentRouterError as exc:
                print(f"[errore] Impossibile riconoscere i parametri: {exc}")
    audio_path = responder.synthesize(response_text)
    if intent_label is not None:
        print(f"Intent rilevato: {intent_label}")
        if slot_annotation:
            print(f"Annotazione parametri: {slot_annotation}")
        print(f"Parametri riconosciuti: {_summarize_slots(slot_values)}")
    print(f"Risposta generata: {response_text}")
    print(f"File audio creato: {audio_path}")

    analyzer = create_analyzer_for_file(str(audio_path), config)
    playback_thread: threading.Thread | None = None


    if not args.no_playback:
        playback_thread = threading.Thread(
            target=playback_audio_file,
            args=(str(audio_path),),
            daemon=True,
        )

    def launch_playback() -> None:
        if playback_thread is not None and not playback_thread.is_alive():
            playback_thread.start()
    try:
        return run_visualizer(
            analyzer,
            window_title="Risposta vocale",
            on_start=launch_playback if playback_thread is not None else None,
        )
    finally:
        if playback_thread is not None:
            playback_thread.join()


def run_listen_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    responses: dict[str, str] = {}
    if args.responses_json:
        responses = load_responses(args.responses_json)

    config = AudioConfig(blocksize=args.blocksize, loop_track=False)

    engine = Pyttsx3Engine(voice=args.voice, rate=args.rate)
    responder = TextToSpeechResponder(
        engine=engine,
        responses=responses,
        output_dir=args.output_dir,
    )

    device: int | str | None
    if args.device is None:
        device = None
    else:
        try:
            device = int(args.device)
        except ValueError:
            device = args.device

    microphone = MicrophoneSource(samplerate=args.samplerate, device=device)
    interactive_source = InteractiveAudioSource(microphone)
    analyzer = AudioAnalyzer(config, source=interactive_source)

    app = create_qt_application()
    window = SphereVisualizer(analyzer)
    window.resize(args.window_size, args.window_size)
    window.setWindowTitle("Assistente vocale")
    window.show()

    listener_queue = microphone.register_listener(maxsize=128)
    recognition_config = RecognitionConfig(
        language=args.language,
        start_threshold=args.start_threshold,
        stop_threshold=args.stop_threshold,
        silence_duration=args.silence_duration,
        min_phrase_duration=args.min_phrase_duration,
        max_phrase_duration=args.phrase_time_limit,
    )
    recognizer = BackgroundRecognizer(listener_queue, microphone.samplerate, config=recognition_config)

    state = {"busy": False, "router_disabled": False}
    router: IntentRouter | None = None

    def handle_text(text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return

        if state["busy"]:
            print("[assistente] Attendere la conclusione della risposta precedente.")
            return

        state["busy"] = True
        recognizer.set_paused(True)

        def worker() -> None:
            nonlocal router
            try:
                print(f"Utente: {cleaned}")
                response_text, matched = responder.find_response(cleaned)
                intent_label: str | None = None
                slot_annotation: str | None = None
                slot_values: dict[str, list[str]] | None = None

                if not matched and not state["router_disabled"]:
                    if router is None:
                        router = IntentRouter(model_path=args.model_path)
                    try:
                        intent_label, response_text = router.describe(cleaned)
                    except IntentRouterError as exc:
                        print(f"[errore] Impossibile analizzare la richiesta: {exc}")
                    else:
                        try:
                            slot_annotation, slot_values = get_annot_utt(cleaned)
                        except IntentRouterError as exc:
                            print(f"[errore] Impossibile riconoscere i parametri: {exc}")
                        router = None
                        state["router_disabled"] = True
                audio_path = responder.synthesize(response_text)
                print(f"Assistente: {response_text}")
                if intent_label is not None:
                    print(f"Intent riconosciuto: {intent_label}")
                    if slot_annotation:
                        print(f"Annotazione parametri: {slot_annotation}")
                    print(f"Parametri riconosciuti: {_summarize_slots(slot_values)}")
                interactive_source.queue_playback_file(audio_path)
                window.set_reactive(True)
                playback_audio_file(str(audio_path))
            except Exception as exc:  # pragma: no cover - errori di runtime in thread
                print(f"[errore] Impossibile completare la risposta: {exc}")
            finally:
                recognizer.set_paused(False)
                window.set_reactive(False)
                state["busy"] = False

        threading.Thread(target=worker, daemon=True).start()

    recognizer.on_text.append(handle_text)
    recognizer.on_error.append(lambda message: print(f"[riconoscimento] {message}"))

    print("Sfera attiva. Parla per ricevere una risposta.")

    analyzer.start()
    recognizer.start()

    try:
        return app.exec_()
    finally:
        recognizer.stop()
        analyzer.stop()
        engine.shutdown()

def run_monitor_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    config = AudioConfig(
        blocksize=args.blocksize,
        use_output_loopback=True,
        loopback_device=args.device or "",
    )
    analyzer = AudioAnalyzer(config)
    return run_visualizer(
        analyzer,
        window_title="Monitor uscita audio",
    )


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        argv = ["listen"]

    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
