"""
genre_map.py
------------
Defines the mapping from the ~84 raw, highly granular genre tags found in the
`listed_in` column down to a small set of broad, learnable genre classes.

The raw dataset lists genres like "TV Dramas", "Dramas", "International TV
Shows", "Action-Adventure", "Action & Adventure", etc. Many of these are the
same underlying genre split only by movie/TV-show naming conventions. For a
genre prediction task to be meaningful (and for classes to have enough
samples to be learnable), we consolidate them into broad buckets.
"""

# Ordered so more specific keywords are checked before generic ones.
GENRE_KEYWORD_MAP = [
    ("Documentary",        ["documentary", "documentaries", "docuseries", "science & nature",
                             "biographical", "faith & spirituality"]),
    ("Kids & Family",      ["kids", "children & family", "family"]),
    ("Anime & Animation",  ["anime", "animation", "cartoon"]),
    ("Horror & Thriller",  ["horror", "thriller", "mystery", "mysteries"]),
    ("Crime",              ["crime"]),
    ("Action & Adventure", ["action"]),
    ("Sci-Fi & Fantasy",   ["sci-fi", "science fiction", "fantasy"]),
    ("Romance",            ["romantic", "romance"]),
    ("Comedy",              ["comedy", "comedies", "stand-up"]),
    ("Reality & Talk Show", ["reality", "talk show"]),
    ("International",      ["international", "spanish-language", "korean", "british",
                             "world"]),
    ("Drama",               ["drama"]),
]

DEFAULT_GENRE = "Other"


def map_to_broad_genre(raw_genre: str) -> str:
    """Map a single raw genre string (e.g. 'TV Dramas') to a broad genre class."""
    g = raw_genre.lower()
    for broad_class, keywords in GENRE_KEYWORD_MAP:
        if any(kw in g for kw in keywords):
            return broad_class
    return DEFAULT_GENRE


def get_primary_broad_genre(listed_in: str) -> str:
    """
    Given the raw `listed_in` cell (comma-separated list of genre tags),
    take the FIRST listed genre (the dataset's primary/most prominent tag)
    and map it to a broad class.
    """
    first_raw_genre = listed_in.split(",")[0].strip()
    return map_to_broad_genre(first_raw_genre)
