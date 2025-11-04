# Assistente

Questa repository fornisce una visualizzazione 3D di una sfera di punti che reagisce a un flusso audio.
L'architettura è stata suddivisa in moduli specializzati per rendere semplice collegare nuove sorgenti audio,
integrare la sintesi vocale e addestrare un modello basato sul dataset MASSIVE di Amazon Science (`massive_dataset`).

## Struttura del progetto

- `assistente/audio/`: componenti per la configurazione dell'analizzatore audio, comprese le sorgenti basate su file.
- `assistente/visualization/`: widget Qt responsabili del rendering e dell'animazione della sfera.
- `assistente/tts/`: motori e utilità per generare risposte vocali da testo.
- `assistente/training/`: script e pipeline di training basati sui dati nella cartella `massive_dataset`
  (è ancora accettata la directory legacy `zendod_dataset/`).
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

## Training con il dataset MASSIVE

La cartella `massive_dataset/` contiene i file JSON Lines del dataset MASSIVE (Amazon Science).
Se stai aggiornando da una versione precedente puoi lasciare i file nella cartella legacy `zendod_dataset/`,
ancora supportata automaticamente dal codice.
Ogni file è nominato come `<configurazione>_<split>.json` (ad es. `massive_train.json`) e include gli utterance (`utt`)
e l'intent associato (`intent`).

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



### Descrizione del dataset MASSIVE

Il **MASSIVE dataset** (*Multilingual Amazon SLU System for Intent Verification and Entity Extraction*) è
un corpus pubblicato da [Amazon Science](https://www.amazon.science/publications/massive-a-1m-example-multilingual-natural-language-understanding-dataset-with-51-languages)
per la ricerca e lo sviluppo di sistemi di comprensione del linguaggio naturale multilingue.

Il dataset contiene **oltre 1 milione di esempi** in **51 lingue**, ciascuno costituito da:
- un’**utterance** (frase o comando vocale),
- un’etichetta di **intent** (intenzione comunicativa),
- e un’annotazione opzionale di **slot** o **parametri** (entità rilevanti estratte dal testo).

Ogni esempio segue la struttura:

```json
{
  "id": "it-00001234",
  "locale": "it-IT",
  "partition": "train",
  "scenario": "music",
  "intent": "PlayMusic",
  "utt": "riproduci musica jazz",
  "annot_utt": "riproduci musica [music_genre : jazz]"
}
```

### Parametri supportati per ogni intent

La tabella seguente riassume i tipi di parametro (slot) osservati per ogni
intent analizzando i file `massive_*` presenti nella cartella `MASSIVE_dataset`
(`annot_utt`). Se un intent non ha ancora parametri annotati la colonna mostra
il simbolo `—`.

| Intent | Parametri (slot) |
| --- | --- |
| `alarm_query` | — |
| `alarm_remove` | — |
| `alarm_set` | — |
| `audio_volume_down` | — |
| `audio_volume_mute` | — |
| `audio_volume_other` | — |
| `audio_volume_up` | — |
| `calendar_query` | — |
| `calendar_remove` | — |
| `calendar_set` | — |
| `cooking_query` | — |
| `cooking_recipe` | — |
| `datetime_convert` | — |
| `datetime_query` | — |
| `email_addcontact` | — |
| `email_query` | — |
| `email_querycontact` | — |
| `email_sendemail` | — |
| `general_greet` | — |
| `general_joke` | — |
| `general_quirky` | — |
| `iot_cleaning` | — |
| `iot_coffee` | — |
| `iot_hue_lightchange` | — |
| `iot_hue_lightdim` | — |
| `iot_hue_lightoff` | — |
| `iot_hue_lighton` | — |
| `iot_hue_lightup` | — |
| `iot_wemo_off` | — |
| `iot_wemo_on` | — |
| `lists_createoradd` | — |
| `lists_query` | — |
| `lists_remove` | — |
| `music_dislikeness` | — |
| `music_likeness` | — |
| `music_query` | — |
| `music_settings` | — |
| `news_query` | — |
| `play_audiobook` | — |
| `play_game` | — |
| `play_music` | — |
| `play_podcasts` | — |
| `play_radio` | — |
| `qa_currency` | — |
| `qa_definition` | — |
| `qa_factoid` | — |
| `qa_maths` | — |
| `qa_stock` | — |
| `recommendation_events` | — |
| `recommendation_locations` | — |
| `recommendation_movies` | — |
| `social_post` | — |
| `social_query` | — |
| `takeaway_order` | — |
| `takeaway_query` | — |
| `transport_query` | — |
| `transport_taxi` | — |
| `transport_ticket` | — |
| `transport_traffic` | — |
| `weather_query` | — |
