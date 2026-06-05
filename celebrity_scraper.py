"""
Downloads celebrity face images via DuckDuckGo image search.
Images are saved to celebrity_data/<Celebrity_Name>/ and used as extra training data.

Install: pip install duckduckgo-search requests
"""

import os
import time
import itertools
import random
import logging
import requests
from PIL import Image

CELEBRITIES = [
    # Hollywood Actors & Actresses
    "Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Cate Blanchett",
    "Denzel Washington", "Natalie Portman", "Brad Pitt", "Angelina Jolie",
    "Will Smith", "Sandra Bullock", "Robert Downey Jr", "Scarlett Johansson",
    "Johnny Depp", "Jennifer Lawrence", "Matt Damon", "Anne Hathaway",
    "Ryan Reynolds", "Emma Stone", "Chris Evans actor", "Margot Robbie",
    "Chris Hemsworth", "Charlize Theron", "Mark Ruffalo", "Reese Witherspoon",
    "Hugh Jackman", "Julia Roberts", "Christian Bale", "Halle Berry",
    "Ben Affleck", "Nicole Kidman", "Jake Gyllenhaal", "Joaquin Phoenix",
    "Viola Davis", "Adam Driver", "Lupita Nyongo", "Tom Hardy",
    "Zoe Saldana", "Javier Bardem", "Penelope Cruz", "Daniel Craig",
    "Rachel McAdams", "Idris Elba", "Naomi Watts", "Michael Fassbender",
    "Kirsten Dunst", "Vin Diesel", "Dwayne Johnson", "Keanu Reeves",
    "Uma Thurman", "Jason Statham", "Mila Kunis", "Ryan Gosling",
    "Eva Mendes", "Jeremy Renner", "Brie Larson", "Oscar Isaac",
    "Daisy Ridley", "John Boyega", "Gal Gadot", "Timothee Chalamet",
    "Florence Pugh", "Zendaya", "Tom Holland", "Ana de Armas",
    "Henry Cavill", "Millie Bobby Brown", "Kit Harington", "Emilia Clarke",
    "Pedro Pascal", "Anya Taylor-Joy", "Paul Mescal", "Sydney Sweeney",
    "Austin Butler", "Jacob Elordi", "Barry Keoghan", "Cillian Murphy",
    "Andrew Garfield", "Jesse Eisenberg", "Taron Egerton", "Richard Madden",
    "Jamie Dornan", "Sam Claflin", "Eddie Redmayne", "Benedict Cumberbatch",
    "James McAvoy", "Colin Firth", "Gary Oldman", "Ian McKellen",
    "Anthony Hopkins", "Judi Dench", "Helena Bonham Carter", "Keira Knightley",
    "Emily Blunt", "Kate Winslet", "Saoirse Ronan", "Olivia Colman",
    "Claire Foy", "Carey Mulligan", "Felicity Jones", "Gemma Arterton",
    "Rosamund Pike", "Rebecca Ferguson", "Michelle Williams actress",
    "Marion Cotillard", "Lea Seydoux", "Audrey Tautou", "Isabelle Huppert",
    "Eva Green", "Ezra Miller actor",

    # Musicians
    "Taylor Swift", "Beyonce", "Jay-Z", "Rihanna", "Ed Sheeran",
    "Adele", "Bruno Mars", "Lady Gaga", "Katy Perry", "Justin Bieber",
    "Ariana Grande", "Billie Eilish", "The Weeknd", "Drake", "Kanye West",
    "Eminem", "Nicki Minaj", "Cardi B", "Post Malone", "Dua Lipa",
    "Harry Styles", "Olivia Rodrigo", "Doja Cat", "Lizzo", "SZA",
    "Bad Bunny", "J Balvin", "Maluma", "Ozuna singer", "Daddy Yankee",
    "Shakira", "Jennifer Lopez", "Marc Anthony", "Pitbull rapper",
    "Camila Cabello", "Selena Gomez", "Miley Cyrus", "Halsey",
    "Lana Del Rey", "Billie Joe Armstrong", "Pink singer", "Alicia Keys",
    "John Legend", "Sam Smith singer", "Shawn Mendes", "Charlie Puth",
    "Niall Horan", "Louis Tomlinson", "Liam Payne", "Zayn Malik",
    "Coldplay Chris Martin", "Muse Matt Bellamy", "Radiohead Thom Yorke",
    "Arctic Monkeys Alex Turner", "Oasis Liam Gallagher", "Oasis Noel Gallagher",
    "Elton John", "Paul McCartney", "Mick Jagger", "Keith Richards",
    "Roger Waters", "David Gilmour", "Robert Plant", "Jimmy Page",
    "Ozzy Osbourne", "Bruce Springsteen", "Bob Dylan", "Neil Young",
    "Eric Clapton", "Stevie Wonder", "Mariah Carey", "Whitney Houston",
    "Janet Jackson", "Michael Jackson", "Prince musician", "David Bowie",
    "Freddie Mercury", "Elvis Presley", "Frank Sinatra", "Dolly Parton",
    "Johnny Cash", "Willie Nelson", "Kenny Rogers", "Garth Brooks",
    "Luke Bryan", "Blake Shelton", "Carrie Underwood", "Miranda Lambert",
    "Kacey Musgraves", "Lil Wayne", "Kendrick Lamar", "J Cole",
    "Travis Scott rapper", "Megan Thee Stallion", "21 Savage", "Roddy Ricch",
    "Polo G", "Jack Harlow", "Gunna rapper", "Future rapper",
    "Lil Baby", "Tyler The Creator", "Chance The Rapper", "Logic rapper",
    "Machine Gun Kelly", "Juice WRLD", "Lil Uzi Vert",

    # K-Pop Stars
    "BTS Jin", "BTS Suga", "BTS J-Hope", "BTS RM", "BTS Jimin",
    "BTS V", "BTS Jungkook", "Blackpink Jennie", "Blackpink Lisa",
    "Blackpink Rose", "Blackpink Jisoo", "EXO Kai", "EXO Sehun",
    "EXO Baekhyun", "GOT7 Jackson Wang", "Stray Kids Bang Chan",
    "Stray Kids Felix", "TWICE Nayeon", "TWICE Sana", "TWICE Momo",
    "NCT Taeyong", "NCT Jaehyun", "IU singer Korean", "CL 2NE1",
    "G-Dragon", "Taeyang BIGBANG", "Psy Gangnam Style", "Aespa Karina",
    "NewJeans Minji", "LE SSERAFIM Kazuha", "Seventeen Mingyu",
    "Ateez Hongjoong", "TXT Yeonjun",

    # Bollywood Stars
    "Shah Rukh Khan", "Salman Khan actor", "Aamir Khan actor",
    "Amitabh Bachchan", "Akshay Kumar", "Hrithik Roshan",
    "Ranveer Singh actor", "Ranbir Kapoor", "Deepika Padukone",
    "Priyanka Chopra", "Alia Bhatt", "Katrina Kaif", "Kareena Kapoor",
    "Aishwarya Rai", "Anushka Sharma actress", "Vidya Balan",
    "Taapsee Pannu", "Kangana Ranaut", "Sonam Kapoor", "Disha Patani",
    "Tiger Shroff", "Varun Dhawan", "Kartik Aaryan", "Ayushmann Khurrana",
    "Rajkummar Rao",

    # Athletes
    "Cristiano Ronaldo", "Lionel Messi", "Neymar", "Kylian Mbappe",
    "LeBron James", "Stephen Curry", "Kevin Durant", "Giannis Antetokounmpo",
    "Luka Doncic", "Nikola Jokic", "Tiger Woods", "Roger Federer",
    "Rafael Nadal", "Novak Djokovic", "Serena Williams", "Naomi Osaka",
    "Maria Sharapova", "Simone Biles", "Usain Bolt", "Carl Lewis",
    "Michael Phelps", "Katie Ledecky", "Tom Brady", "Patrick Mahomes",
    "Aaron Rodgers", "Peyton Manning", "Kobe Bryant", "Michael Jordan",
    "Shaquille O'Neal", "Magic Johnson", "Larry Bird", "Wayne Gretzky",
    "Sidney Crosby", "Alexander Ovechkin", "Mike Tyson", "Muhammad Ali",
    "Floyd Mayweather", "Conor McGregor", "Ronda Rousey", "Canelo Alvarez",
    "Anthony Joshua", "Tyson Fury", "Zlatan Ibrahimovic", "Ronaldinho",
    "Thierry Henry", "David Beckham", "Zinedine Zidane", "Pele footballer",
    "Virgil van Dijk", "Mohamed Salah", "Harry Kane", "Kevin De Bruyne",
    "Erling Haaland", "Robert Lewandowski", "Karim Benzema",
    "Lamine Yamal", "Vinicius Junior",

    # TV Personalities & Hosts
    "Oprah Winfrey", "Ellen DeGeneres", "Jimmy Fallon", "Jimmy Kimmel",
    "Stephen Colbert", "John Oliver", "Trevor Noah", "Seth Meyers",
    "James Corden", "Conan O'Brien", "Jay Leno", "David Letterman",
    "Craig Ferguson", "Graham Norton", "Kelly Ripa", "Ryan Seacrest",
    "Nick Cannon", "Steve Harvey", "Tyra Banks", "Heidi Klum",
    "Gordon Ramsay", "Anthony Bourdain", "Guy Fieri", "Martha Stewart",
    "Phil McGraw Dr Phil", "Anderson Cooper", "Wolf Blitzer",
    "Rachel Maddow", "Robin Roberts", "Diane Sawyer", "Christiane Amanpour",

    # Comedians
    "Dave Chappelle", "Chris Rock", "Kevin Hart", "Jerry Seinfeld",
    "Bill Burr", "Amy Schumer", "Tina Fey", "Amy Poehler",
    "Mindy Kaling", "John Mulaney", "Bo Burnham", "Hannah Gadsby",
    "Wanda Sykes", "Hasan Minhaj", "Jim Gaffigan", "Sebastian Maniscalco",
    "Gabriel Iglesias", "Russell Brand", "Ricky Gervais", "John Cleese",
    "Rowan Atkinson", "Steve Martin", "Eddie Murphy", "Richard Pryor",
    "Robin Williams", "Billy Crystal", "Adam Sandler", "Jim Carrey",
    "Steve Carell", "Will Ferrell", "Zach Galifianakis", "Jonah Hill",
    "Seth Rogen", "Aziz Ansari",

    # Politicians & Business Leaders
    "Barack Obama", "Michelle Obama", "Joe Biden", "Donald Trump",
    "Hillary Clinton", "Bill Clinton", "Kamala Harris", "Bernie Sanders",
    "Alexandria Ocasio-Cortez", "Justin Trudeau", "Emmanuel Macron",
    "Angela Merkel", "Boris Johnson", "Rishi Sunak", "Jacinda Ardern",
    "Volodymyr Zelensky", "Elon Musk", "Jeff Bezos", "Bill Gates",
    "Mark Zuckerberg", "Tim Cook Apple", "Sundar Pichai", "Warren Buffett",

    # YouTubers & Streamers
    "PewDiePie", "MrBeast", "Logan Paul", "Jake Paul", "KSI",
    "Markiplier", "Jacksepticeye", "Ninja streamer", "Pokimane",
    "Kai Cenat", "IShowSpeed", "Emma Chamberlain", "David Dobrik",
    "Addison Rae", "Charli D'Amelio", "Khaby Lame", "Liza Koshy",
    "Casey Neistat", "Mark Rober",

    # Reality TV & Models
    "Kim Kardashian", "Khloe Kardashian", "Kylie Jenner", "Kendall Jenner",
    "Kris Jenner", "Paris Hilton", "Britney Spears", "Lindsay Lohan",
    "Naomi Campbell", "Cindy Crawford", "Claudia Schiffer", "Kate Moss",
    "Gisele Bundchen", "Adriana Lima", "Bella Hadid", "Gigi Hadid",
    "Cara Delevingne", "Ashley Graham", "Karlie Kloss", "Emily Ratajkowski",
    "Miranda Kerr",

    # International Cinema
    "Antonio Banderas", "Salma Hayek", "Gael Garcia Bernal",
    "Monica Bellucci", "Roberto Benigni", "Juliette Binoche",
    "Vincent Cassel", "Catherine Deneuve", "Tony Leung", "Jackie Chan",
    "Jet Li", "Michelle Yeoh", "Zhang Ziyi", "Gong Li",
    "Ken Watanabe", "Song Kang-ho", "Lee Byung-hun", "Hyun Bin",
    "Son Ye-jin",

    # Royals
    "Prince William", "Princess Kate Middleton", "Prince Harry",
    "Meghan Markle", "Princess Diana", "Queen Elizabeth II", "King Charles III",

    # Historical & Notable Figures
    "Nelson Mandela", "Martin Luther King Jr", "Mahatma Gandhi",
    "Dalai Lama", "Pope Francis", "Stephen Hawking", "Neil deGrasse Tyson",
    "Bill Nye", "Malala Yousafzai", "Greta Thunberg", "Joe Rogan",
    "Audrey Hepburn", "Marilyn Monroe", "James Dean", "Marlon Brando",
    "Elvis Presley", "Frida Kahlo", "Andy Warhol", "Steve Jobs",
]

SCRAPE_ROOT = "celebrity_data"
_IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.webp'}


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder)
               if os.path.splitext(f)[1].lower() in _IMG_EXTS)


_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}


def scrape_celebrity(name, n=25, root=SCRAPE_ROOT):
    """
    Download up to n face images for one celebrity via DuckDuckGo image search.
    Returns the number of images now stored in the folder.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError(
            "ddgs not installed — run: pip install ddgs requests")

    safe   = name.replace(' ', '_')
    outdir = os.path.join(root, safe)
    os.makedirs(outdir, exist_ok=True)

    if count_images(outdir) >= n:
        return count_images(outdir)

    # Silence requests noise
    logging.getLogger('urllib3').setLevel(logging.CRITICAL)

    existing = count_images(outdir)
    idx      = existing

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(
                f'{name} face portrait photo',
                max_results=n * 3,   # fetch extra since some URLs will fail
                type_image='photo',
            ))
    except Exception:
        return count_images(outdir)

    for r in results:
        if count_images(outdir) >= n:
            break
        url = r.get('image', '')
        if not url:
            continue
        try:
            # Guess extension from URL, default to .jpg
            raw_path = url.split('?')[0]
            ext = os.path.splitext(raw_path)[1].lower()
            if ext not in _IMG_EXTS:
                ext = '.jpg'
            idx += 1
            dest = os.path.join(outdir, f'{idx:04d}{ext}')
            if os.path.exists(dest):
                continue
            resp = requests.get(url, timeout=8, headers=_HEADERS)
            if resp.status_code == 200 and len(resp.content) > 8_000:
                with open(dest, 'wb') as f:
                    f.write(resp.content)
        except Exception:
            continue

    return count_images(outdir)


def scrape_all(celebrities=None, n_per_celebrity=25, progress_cb=None):
    """
    Download images for every celebrity in the list.
    progress_cb(name: str, done: int, total: int) is called after each celebrity.
    Returns total images stored across all folders.
    """
    if celebrities is None:
        celebrities = CELEBRITIES
    total = len(celebrities)
    grand = 0
    for i, name in enumerate(celebrities):
        grand += scrape_celebrity(name, n=n_per_celebrity)
        if progress_cb:
            progress_cb(name, i + 1, total)
        time.sleep(1.5)   # avoid DuckDuckGo rate limiting
    return grand


_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _groq_client(api_key):
    """Return a Groq client, raising RuntimeError if groq isn't installed."""
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError:
        raise RuntimeError("groq not installed — run: pip install groq")


def _image_b64(image_path):
    import base64
    ext  = os.path.splitext(image_path)[1].lower().lstrip('.')
    mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'png': 'image/png',  'webp': 'image/webp'}.get(ext, 'image/jpeg')
    with open(image_path, 'rb') as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def verify_with_groq(image_path, celebrity_name, client):
    """
    Ask Groq vision whether the image shows the named celebrity.
    Returns True (keep) or False (delete). Raises on API errors so caller can log them.
    Pass a Groq client instance (not api_key) — create once with _groq_client().
    """
    resp = client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": _image_b64(image_path)}},
                {"type": "text",
                 "text": (f"Is this a real photograph clearly showing the face of "
                          f"{celebrity_name}? Reply with only 'yes' or 'no'.")},
            ],
        }],
        max_tokens=3,
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower().startswith('y')


def groq_identify_face(image_pil, api_key):
    """
    Ask Groq vision who the person in image_pil is.
    Returns a short identification string, or an error message.
    """
    import base64, io
    buf = io.BytesIO()
    image_pil.save(buf, format='JPEG', quality=85)
    b64_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    try:
        client = _groq_client(api_key)
        resp   = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": b64_url}},
                    {"type": "text",
                     "text": ("Who is the person in this photo? "
                              "If you recognise them give their full name and what they are known for (actor, musician, etc.). "
                              "If you don't recognise them say 'Unknown person'. "
                              "One sentence, be concise.")},
                ],
            }],
            max_tokens=80,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except RuntimeError as e:
        return str(e)
    except Exception as e:
        return f"AI error: {e}"


def ai_verify_all(root=SCRAPE_ROOT, api_key=None, progress_cb=None):
    """
    Run Groq AI verification on all downloaded images.
    Deletes images where the AI says the wrong/no celebrity is shown.
    Returns (kept, removed).
    Raises RuntimeError if groq isn't installed or the key is invalid.
    """
    if not api_key:
        return 0, 0

    # Validate key with a cheap text call before processing thousands of images
    client = _groq_client(api_key)
    try:
        client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
    except Exception as e:
        raise RuntimeError(f"Groq key test failed: {e}")

    all_items = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        celebrity_name = name.replace('_', ' ')
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                all_items.append((os.path.join(folder, f), celebrity_name))

    kept = removed = errors = 0
    for i, (path, celeb) in enumerate(all_items):
        try:
            keep = verify_with_groq(path, celeb, client)
        except Exception:
            keep  = True   # skip on transient network errors
            errors += 1
        if keep:
            kept += 1
        else:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
        if progress_cb:
            progress_cb(i + 1, len(all_items), kept, removed)
        time.sleep(0.15)

    return kept, removed


def clean_non_faces(root=SCRAPE_ROOT, progress_cb=None):
    """
    Delete downloaded images that don't contain a detectable human face.
    Uses mediapipe — call this after scrape_all() to remove junk images.
    Returns (kept, removed).
    """
    try:
        from face_features import _get_mesh, _HAVE_MP
        import numpy as np
    except ImportError:
        return 0, 0

    if not _HAVE_MP:
        return 0, 0

    all_files = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                all_files.append(os.path.join(folder, f))

    kept = removed = 0
    mesh = _get_mesh()

    for i, path in enumerate(all_files):
        has_face = False
        try:
            img_np = np.array(Image.open(path).convert('RGB'))
            result = mesh.process(img_np)
            has_face = bool(result.multi_face_landmarks)
        except Exception:
            has_face = False  # corrupt / unreadable — treat as no face → delete

        if has_face:
            kept += 1
        else:
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass

        if progress_cb:
            progress_cb(i + 1, len(all_files), kept, removed)

    return kept, removed


def get_scraped_pairs(root=SCRAPE_ROOT, max_pos=3000):
    """
    Generate (path1, path2, label) triples from the downloaded celebrity images.
    Same folder = same person (label 1.0), different folders = different (label 0.0).
    """
    if not os.path.isdir(root):
        return []

    person_images = {}
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        imgs = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if len(imgs) >= 2:
            person_images[name] = imgs

    if not person_images:
        return []

    # Positive pairs — capped at 10 per person so no celebrity dominates
    all_positive = []
    for name, imgs in person_images.items():
        combos = list(itertools.combinations(imgs, 2))
        if len(combos) > 10:
            combos = random.sample(combos, 10)
        all_positive.extend((p1, p2, 1.0) for p1, p2 in combos)
    random.shuffle(all_positive)
    all_positive = all_positive[:max_pos]

    # Negative pairs — equal number, random cross-person
    flat     = [(n, p) for n, imgs in person_images.items() for p in imgs]
    negative = []
    attempts = 0
    while len(negative) < len(all_positive) and attempts < len(all_positive) * 20:
        (n1, p1), (n2, p2) = random.sample(flat, 2)
        if n1 != n2:
            negative.append((p1, p2, 0.0))
        attempts += 1

    return all_positive + negative
