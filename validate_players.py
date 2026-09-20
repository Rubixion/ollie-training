"""Player validator: shows the first image of each player folder. Left/Right arrows to step, Esc to quit."""
import os
import tkinter as tk

from PIL import Image, ImageTk

# ponytail: same folder as SCRAPE_ROOT, hardcoded so we don't import the scraper's heavy deps
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celebrity_data")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SIZE = 600

players = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
if not players:
    raise SystemExit(f"No player folders in {ROOT}")


def first_image(player):
    folder = os.path.join(ROOT, player)
    imgs = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in EXTS)
    return (os.path.join(folder, imgs[0]) if imgs else None), len(imgs)


win = tk.Tk()
win.title("Player validator")
title = tk.Label(win, font=("Segoe UI", 16))
title.pack(pady=8)
pic = tk.Label(win, width=SIZE, height=SIZE)
pic.pack(padx=10, pady=10)
i = 0


def show():
    path, n = first_image(players[i])
    title.config(text=f"{i + 1}/{len(players)}   {players[i].replace('_', ' ')}   ({n} images)")
    pic.image = None
    if path is None:
        pic.config(image="", text="NO IMAGES", fg="red", font=("Segoe UI", 28))
        return
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((SIZE, SIZE))
        pic.image = ImageTk.PhotoImage(img)  # keep a reference or Tk drops it
        pic.config(image=pic.image, text="")
    except Exception as e:
        pic.config(image="", text=f"UNREADABLE\n{os.path.basename(path)}\n{e}", fg="red", font=("Segoe UI", 14))


def step(d):
    global i
    i = max(0, min(len(players) - 1, i + d))
    show()


win.bind("<Right>", lambda e: step(1))
win.bind("<Left>", lambda e: step(-1))
win.bind("<Escape>", lambda e: win.destroy())
show()
win.mainloop()
