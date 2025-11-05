import json
import tkinter as tk
from tkinter import ttk

JSON_FILE = "simulation.json"
REFRESH_MS = 2000  # ogni 2 secondi


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stato casa")

        # frame principali
        self.place_frame = ttk.LabelFrame(self, text="Stanby / luci / place")
        self.place_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.alarms_frame = ttk.LabelFrame(self, text="Allarmi")
        self.alarms_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.music_frame = ttk.LabelFrame(self, text="Musica")
        self.music_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        # bottone refresh manuale
        refresh_btn = ttk.Button(self, text="Aggiorna adesso", command=self.update_ui)
        refresh_btn.grid(row=2, column=0, columnspan=2, pady=5)

        # per tenere i riferimenti ai widget di place
        self.place_widgets = {}
        # per gli allarmi li ricreo ogni volta
        # per la musica tengo le label
        self.music_labels = {}

        # primo caricamento
        self.update_ui()

        # aggiornamento periodico
        self.after(REFRESH_MS, self.auto_refresh)

        # un po' di layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

    def load_data(self):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_dot(self, parent, color, size=12):
        c = tk.Canvas(parent, width=size, height=size, highlightthickness=0)
        c.create_oval(2, 2, size-2, size-2, fill=color, outline=color)
        return c

    def update_place(self, place_dict):
        # cancella tutto e ricrea (più semplice)
        for widget in self.place_frame.winfo_children():
            widget.destroy()

        for r, (room, value) in enumerate(place_dict.items()):
            color = "green" if value else "red"
            dot = self.make_dot(self.place_frame, color)
            dot.grid(row=r, column=0, padx=(5, 10), pady=3, sticky="w")
            ttk.Label(self.place_frame, text=room).grid(row=r, column=1, sticky="w")

    def update_alarms(self, alarms_list):
        # cancella e ricrea
        for widget in self.alarms_frame.winfo_children():
            widget.destroy()

        if alarms_list:
            for i, alarm in enumerate(alarms_list):
                ttk.Label(self.alarms_frame, text=f"• {alarm}").grid(row=i, column=0, sticky="w", padx=5, pady=2)
        else:
            ttk.Label(self.alarms_frame, text="(nessun allarme)").grid(row=0, column=0, sticky="w", padx=5, pady=2)

    def update_music(self, music_dict):
        for widget in self.music_frame.winfo_children():
            widget.destroy()

        brano = music_dict.get("brano")
        autore = music_dict.get("autore")
        player = music_dict.get("player")

        ttk.Label(self.music_frame, text=f"Brano: {brano}").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        row = 1
        if autore is not None:
            ttk.Label(self.music_frame, text=f"Autore: {autore}").grid(row=row, column=0, sticky="w", padx=5, pady=3)
            row += 1
        if player is not None:
            ttk.Label(self.music_frame, text=f"Player: {player}").grid(row=row, column=0, sticky="w", padx=5, pady=3)

    def update_ui(self):
        data = self.load_data()

        self.update_place(data.get("place", {}))
        self.update_alarms(data.get("alarms", []))
        self.update_music(data.get("music", {}))

    def auto_refresh(self):
        # chiamata periodica
        self.update_ui()
        self.after(REFRESH_MS, self.auto_refresh)


if __name__ == "__main__":
    app = App()
    app.mainloop()
