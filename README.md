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

Per analizzare rapidamente la distribuzione degli intent nel dataset puoi utilizzare l'apposito visualizzatore:

```bash
python -m assistente.training.data_visualizer --config massive --split train --no-show
```

L'utility stampa una tabella riassuntiva e, se `matplotlib` è installato, genera anche un grafico (salvabile con `--save-plot`).

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

| Intent                     | Parametri (slot)                                             |
| -------------------------- | ------------------------------------------------------------ |
| `alarm_query`              | `datetime`, `alarm_name`                                     |
| `alarm_remove`             | `datetime`, `alarm_name`                                     |
| `alarm_set`                | `datetime`, `alarm_name`, `recurrence`                       |
| `audio_volume_down`        | `device`, `location`                                         |
| `audio_volume_mute`        | `device`, `location`                                         |
| `audio_volume_other`       | `device`, `location`, `volume_level`                         |
| `audio_volume_up`          | `device`, `location`                                         |
| `calendar_query`           | `datetime`, `event_name`, `location`, `participant`          |
| `calendar_remove`          | `datetime`, `event_name`                                     |
| `calendar_set`             | `datetime`, `event_name`, `location`, `participant`          |
| `cooking_query`            | `dish_name`, `ingredient`, `dietary_preference`              |
| `cooking_recipe`           | `dish_name`, `ingredient`, `dietary_preference`              |
| `datetime_convert`         | `source_datetime`, `target_timezone`                         |
| `datetime_query`           | `datetime`, `location`                                       |
| `email_addcontact`         | `contact_name`, `email_address`                              |
| `email_query`              | `contact_name`, `datetime`, `email_subject`                  |
| `email_querycontact`       | `contact_name`                                               |
| `email_sendemail`          | `contact_name`, `email_subject`, `message_body`              |
| `general_greet`            | —                                                            |
| `general_joke`             | —                                                            |
| `general_quirky`           | —                                                            |
| `iot_cleaning`             | `device`, `location`                                         |
| `iot_coffee`               | `device`, `drink_type`, `size`, `strength`                   |
| `iot_hue_lightchange`      | `device`, `location`, `color`                                |
| `iot_hue_lightdim`         | `device`, `location`, `brightness`                           |
| `iot_hue_lightoff`         | `device`, `location`                                         |
| `iot_hue_lighton`          | `device`, `location`, `color`                                |
| `iot_hue_lightup`          | `device`, `location`, `brightness`                           |
| `iot_wemo_off`             | `device`, `location`                                         |
| `iot_wemo_on`              | `device`, `location`                                         |
| `lists_createoradd`        | `list_name`, `list_item`                                     |
| `lists_query`              | `list_name`, `list_item`                                     |
| `lists_remove`             | `list_name`, `list_item`                                     |
| `music_dislikeness`        | `music_artist`, `music_genre`, `music_name`                  |
| `music_likeness`           | `music_artist`, `music_genre`, `music_name`                  |
| `music_query`              | `music_artist`, `music_genre`, `music_name`                  |
| `music_settings`           | `music_provider`, `device`, `location`                       |
| `news_query`               | `news_category`, `news_source`, `datetime`                   |
| `play_audiobook`           | `book_name`, `book_author`, `chapter`                        |
| `play_game`                | `game_name`, `device`                                        |
| `play_music`               | `music_artist`, `music_genre`, `music_name`, `playlist_name` |
| `play_podcasts`            | `podcast_name`, `podcast_topic`, `episode_number`            |
| `play_radio`               | `radio_name`, `radio_frequency`, `location`                  |
| `qa_currency`              | `currency_name`, `amount`, `target_currency`                 |
| `qa_definition`            | `word`                                                       |
| `qa_factoid`               | `topic`                                                      |
| `qa_maths`                 | `math_expression`                                            |
| `qa_stock`                 | `company_name`                                               |
| `recommendation_events`    | `event_type`, `location`, `datetime`                         |
| `recommendation_locations` | `location`, `poi_type`, `rating`                             |
| `recommendation_movies`    | `movie_name`, `movie_genre`, `actor_name`                    |
| `social_post`              | `platform_name`, `message_body`, `contact_name`              |
| `social_query`             | `platform_name`, `contact_name`                              |
| `takeaway_order`           | `restaurant_name`, `dish_name`, `quantity`, `location`       |
| `takeaway_query`           | `restaurant_name`, `dish_name`, `location`                   |
| `transport_query`          | `origin`, `destination`, `datetime`, `transport_type`        |
| `transport_taxi`           | `origin`, `destination`, `datetime`, `vehicle_type`          |
| `transport_ticket`         | `origin`, `destination`, `datetime`, `transport_type`        |
| `transport_traffic`        | `location`, `datetime`                                       |
| `weather_query`            | `location`, `datetime`                                       |

