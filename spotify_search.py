"""
spotify_search.py
─────────────────────────────────────────────────────────────────────────────
Finds any song and returns a 30-second audio preview URL.

Uses iTunes Search API (Apple Music) as primary source:
  - No API key required, completely free
  - Returns 30-second .m4a previews for millions of tracks
  - Falls back to Spotify embed preview if iTunes has no result

Usage:
    from spotify_search import find_preview
    result = find_preview("Blinding Lights The Weeknd")
    # {"title": "Blinding Lights", "artist": "The Weeknd", "preview_url": "https://..."}
─────────────────────────────────────────────────────────────────────────────
"""

import re
import json
import urllib.parse
import urllib.request
from typing import Optional


ITUNES_SEARCH  = "https://itunes.apple.com/search"
SPOTIFY_EMBED  = "https://open.spotify.com/embed/track/{track_id}"
SPOTIFY_TOKEN  = "https://open.spotify.com/get_access_token?reason=transport&productType=web_player"
SPOTIFY_SEARCH = "https://api.spotify.com/v1/search"
PREVIEW_RE     = re.compile(r'"audioPreview"\s*:\s*\{"url"\s*:\s*"([^"]+)"')


def _itunes_search(query: str) -> Optional[dict]:
    """Search iTunes. Returns track metadata + preview URL or None."""
    params = urllib.parse.urlencode({
        "term":   query,
        "media":  "music",
        "limit":  1,
    })
    req = urllib.request.Request(
        f"{ITUNES_SEARCH}?{params}",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("resultCount", 0) == 0:
            return None
        t = data["results"][0]
        preview = t.get("previewUrl")
        if not preview:
            return None
        return {
            "title":       t.get("trackName", "Unknown"),
            "artist":      t.get("artistName", "Unknown"),
            "preview_url": preview,
            "source":      "itunes",
        }
    except Exception:
        return None


def _spotify_token() -> Optional[str]:
    """Get anonymous Spotify web player token."""
    try:
        req = urllib.request.Request(
            SPOTIFY_TOKEN,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("accessToken")
    except Exception:
        return None


def _spotify_preview(query: str) -> Optional[dict]:
    """Search Spotify and extract preview via embed page."""
    token = _spotify_token()
    if not token:
        return None
    try:
        params = urllib.parse.urlencode({"q": query, "type": "track", "limit": 1})
        req = urllib.request.Request(
            f"{SPOTIFY_SEARCH}?{params}",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        items = data.get("tracks", {}).get("items", [])
        if not items:
            return None
        track      = items[0]
        track_id   = track["id"]
        preview    = track.get("preview_url")

        # If search didn't give preview, try embed page
        if not preview:
            embed_req = urllib.request.Request(
                SPOTIFY_EMBED.format(track_id=track_id),
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(embed_req, timeout=10) as r:
                html = r.read().decode("utf-8", errors="ignore")
            match = PREVIEW_RE.search(html)
            if match:
                preview = match.group(1)

        if not preview:
            return None

        return {
            "title":       track["name"],
            "artist":      track["artists"][0]["name"],
            "preview_url": preview,
            "source":      "spotify",
        }
    except Exception:
        return None


def find_preview(song_query: str) -> dict:
    """
    Find a song and return its 30-second preview URL.

    Args:
        song_query: Song name and/or artist, e.g. "Blinding Lights The Weeknd"

    Returns:
        {"title": str, "artist": str, "preview_url": str, "source": str}

    Raises:
        ValueError if no result or no preview found.
    """
    # Try iTunes first — fastest, no auth needed
    result = _itunes_search(song_query)
    if result:
        return result

    # Fallback: Spotify
    result = _spotify_preview(song_query)
    if result:
        return result

    raise ValueError(
        f"Could not find a playable preview for '{song_query}'. "
        "Try including the artist name, e.g. 'Song Title Artist Name'."
    )
