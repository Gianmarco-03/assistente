# Assistente

Questa repository fornisce una visualizzazione 3D di una sfera di punti che reagisce a un flusso audio.
L'architettura è stata suddivisa in moduli specializzati per rendere semplice collegare nuove sorgenti audio,
integrare la sintesi vocale e addestrare un modello basato sul dataset `zendod_dataset`.

## Struttura del progetto

- `assistente/audio/`: componenti per la configurazione dell'analizzatore audio, comprese le sorgenti basate su file.
- `assistente/visualization/`: widget Qt responsabili del rendering e dell'animazione della sfera.
- `assistente/tts/`: motori e utilità per generare risposte vocali da testo.
- `assistente/training/`: script e pipeline di training basati sui dati nella cartella `zendod_dataset`.
- `assistente/main.py`: interfaccia a riga di comando che unisce visualizzazione, TTS e riproduzione audio.
- `sfera.py`: punto di ingresso legacy che delega al nuovo CLI.

## Requisiti principali

Per utilizzare tutte le funzionalità è necessario installare le seguenti librerie Python:

```bash
pip install pyqt5 pyqtgraph soundfile sounddevice pyttsx3 scikit-learn joblib
```

Alcune funzionalità (come la riproduzione audio diretta o il training) possono funzionare anche senza tutte
le dipendenze installate, ma il programma mostrerà un messaggio d'avviso quando mancano componenti opzionali.

## Utilizzo della CLI

La CLI principale espone due comandi: `play` e `speak`.

Riprodurre un file audio esistente nella sfera:

```bash
python -m assistente.main play path/al/file.wav
```

Generare una risposta vocale a partire da un testo (con animazione e riproduzione del file generato):

```bash
python -m assistente.main speak "ciao"
```

Monitorare l'audio di uscita del sistema (richiede `sounddevice` e un driver compatibile):

```bash
python -m assistente.main monitor
```

Sono disponibili opzioni per personalizzare il blocco di analisi, il dizionario di risposte (`--responses-json`),
la cartella di output e i parametri del motore TTS (`--voice`, `--rate`). Usa `--help` per l'elenco completo.

## Text-to-Speech

Il modulo `assistente/tts/responder.py` definisce un componente che associa input testuali a risposte vocali
predefinite. Il testo prodotto viene sintetizzato tramite `pyttsx3` e salvato in un file WAV nella cartella `tts_output`
(di default). Il file viene quindi riprodotto (se `sounddevice` è installato) e passato all'animazione della sfera.

## Training con `zendod_dataset`

La cartella `zendod_dataset/` contiene i file JSON lines del dataset ITALIC. Ogni file è nominato come
`<configurazione>_<split>.json` (ad es. `massive_train.json`) e include gli utterance (`utt`) e l'intent associato
(`intent`).

Per avviare l'addestramento e generare un modello `joblib` pronto all'uso, assicurati di aver estratto l'intero dataset e poi
esegui:


```bash
python -m assistente.training.train --config massive --train-split train --eval-split validation
```

Puoi cambiare `--config` per utilizzare una configurazione diversa (es. `hard_noisy` o `hard_speaker`) e modificare gli split
se vuoi valutare su `test` o creare una suddivisione casuale interna allo split di training (omettendo `--eval-split`).

Il modello verrà salvato nella cartella `models/` (configurabile con `--output-dir`) e l'output del comando mostrerà
un report di valutazione basato sullo split scelto. Questo modello può essere utilizzato per arricchire le risposte vocali e
aggiornare le mappature di `TextToSpeechResponder`.

## Licenza

Questo progetto è fornito così com'è per scopi dimostrativi ed educativi.