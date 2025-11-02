"""Interfaccia a riga di comando per il progetto Assistente."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from assistente.app import create_analyzer_for_file, playback_audio_file, run_visualizer
from assistente.audio.config import AudioConfig
from assistente.tts.pyttsx3_engine import Pyttsx3Engine
from assistente.tts.responder import TextToSpeechResponder

DEFAULT_RESPONSES: Dict[str, str] = {
    "ciao": "Ciao! Sono qui per aiutarti con la visualizzazione della sfera.",
    "come stai": "Sto benissimo, grazie! Pronta a generare nuove animazioni.",
    "aiuto": "Dimmi pure come posso assisterti con la sfera interattiva.",
}


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
    speak_parser.set_defaults(func=run_speak_command)

    return parser


def run_play_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    config = AudioConfig(blocksize=args.blocksize, loop_track=not args.no_loop)
    analyzer = create_analyzer_for_file(args.audio_path, config)
    return run_visualizer(analyzer)


def load_responses(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SystemExit("Il file JSON deve contenere un dizionario 'input' -> 'response'.")
    return {str(key): str(value) for key, value in data.items()}


def run_speak_command(args: argparse.Namespace) -> int:
    if args.blocksize <= 0:
        raise SystemExit("Il parametro --blocksize deve essere un intero positivo.")

    responses = DEFAULT_RESPONSES.copy()
    if args.responses_json:
        responses.update(load_responses(args.responses_json))

    config = AudioConfig(blocksize=args.blocksize, loop_track=False)

    engine = Pyttsx3Engine(voice=args.voice, rate=args.rate)
    responder = TextToSpeechResponder(
        engine=engine,
        responses=responses,
        output_dir=args.output_dir,
    )

    response_text, audio_path = responder.respond(args.text)
    print(f"Risposta generata: {response_text}")
    print(f"File audio creato: {audio_path}")

    if not args.no_playback:
        playback_audio_file(str(audio_path))

    analyzer = create_analyzer_for_file(str(audio_path), config)
    try:
        return run_visualizer(analyzer, window_title="Risposta vocale")
    finally:
        engine.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
