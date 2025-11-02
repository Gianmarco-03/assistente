# Dataset "zendod_dataset"

Inserisci in questa cartella il file `responses.csv` con le colonne `prompt` e `response`.
Ogni riga rappresenta una coppia testo-domanda e risposta attesa utilizzata per addestrare il
modello di classificazione presente in `assistente/training/`.

Esempio di struttura:

```csv
prompt,response
ciao,Ciao! Sono la tua assistente virtuale.
come stai,Sto benissimo, grazie!
```

Puoi aggiungere ulteriori colonne per meta-informazioni, ma solo `prompt` e `response` vengono utilizzate
dal pipeline di training.
