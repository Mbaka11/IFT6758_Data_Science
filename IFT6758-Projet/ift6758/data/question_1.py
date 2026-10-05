import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


class NHLDataRetriever: 
    
    BASE_URL = "https://api-web.nhle.com/v1"
    GAME_LIST_URL = "https://api.nhle.com/stats/rest/en/game"

    def __init__(
        self, 
        data_dir: str | Path | None = None,
        *,
        project_root: str | Path,
        timeout: float = 30.0,
        min_request_interval: float = 1.0,
    ) -> None:
        """Resolve the storage path and retain validated request settings.
        
        Args:
            data_dir: Explicit storage path, taking priority over NHL_DATA_DIR.
            project_root: Path to the existing project directory.
            timeout: Positive request timeout in seconds.
            min_request_interval: Nonnegative spacing between HTTP requests.

        Raises:
            TypeError: A path or setting has the wrong type.
            ValueError: The project root, data path, or setting is invalid.
        """
        root = Path(project_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("project_root must be an existing directory")

        # Prefer an explicit path, then the .env value, then the default.
        selected_path = (
            data_dir
            if data_dir is not None
            else os.environ.get("NHL_DATA_DIR", "data/raw/nhl")
        )
        if isinstance(selected_path, str) and not selected_path.strip():
            raise ValueError("data_dir must not be blank")

        storage = Path(selected_path).expanduser()
        if not storage.is_absolute():
            storage = root / storage

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if min_request_interval < 0:
            raise ValueError("min_request_interval cannot be negative")

        self.project_root = root
        self.data_dir = storage.resolve()
        self.timeout = timeout
        self.min_request_interval = min_request_interval
        self._last_request_started: float | None = None
    
    def get_game(self, game_id: int) -> dict:
        """Return one game's complete raw JSON, loading the cache first.
        
        Args:
            game_id: Ten digit integer, example: 2023021190
            
        Returns:
            A dictionary containing the game's raw JSON data.
        
        Raises:
            TypeError: The game_id is not an integer.
            ValueError: The game_id, JSON data, or basic game structure is invalid.
            requests.RequestException: The HTTP request failed or returned an error status.
            OSError: The cache file could not be read or written.
        """
        
        if type(game_id) is not int:
            raise TypeError("game_id must be an integer")
        if not 1_000_000_000 <= game_id <= 9_999_999_999:
            raise ValueError("game_id must contain ten digits")

        # Keep each game's raw JSON in its season folder.
        cache_path = self._game_cache_path(game_id)
        from_cache = cache_path.exists()
        # If cache file exists, load from cache, otherwise fetch from API
        if from_cache:
            game = self._load_game(cache_path)
        else:
            game = self._fetch_game(game_id)
            
        # Validate basic structure of the game data, result is a dict, with matching id and has a plays list    
        if (
            not isinstance(game, dict)
            or game.get("id") != game_id
            or not isinstance(game.get("plays"), list)
        ):
            raise ValueError(
                f"Invalid data for game {game_id}: expected matching id and a plays list"
            )

        # Save newly fetched data and return JSON data
        if not from_cache:
            self._save_game(cache_path, game)
        return game

    def _game_cache_path(self, game_id: int) -> Path:
        """Return the season-organized path for one game's raw JSON file.
        
        Args: 
            game_id: Ten digit integer, example: 2023021191
        Returns:
            Path to the JSON file for the game, organized by season folder.
        """
        season_start_year = int(str(game_id)[:4])
        season_folder = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
        return self.data_dir / "games" / season_folder / f"{game_id}.json"

    def get_season_game_ids(
        self, season_start_year: int, *, refresh: bool = False
    ) -> list[int]:
        """Discover completed regular-season/playoff IDs, using cached metadata.
        Doesn't download the play-by-play JSON for each game, just the metadata list of IDs.
        Game types 2 and 3 are regular season and playoffs, respectively. 
        Game states 6 and 7 are ended/final. Unplayed 'if necessary' playoff games are excluded.
        
        Args:
            season_start_year: Starting year, example : 2023 for the 2023–24 season.
            refresh: Explicitly fetch fresh metadata instead of a cached list.

        Returns:
            Sorted, deduplicated IDs for game types 2 and 3 in ended/final states
            6 and 7. Unplayed 'if necessary' playoff games are excluded.

        Raises:
            TypeError: The starting year is not an integer.
            ValueError: The year or metadata is invalid/incomplete.
            requests.RequestException: Metadata retrieval fails.
            OSError: Metadata cache reading/writing fails.
        """
        if type(season_start_year) is not int:
            raise TypeError("season_start_year must be an integer")
        # Restrain to the seasons requiered for project 1, 2016-17 through 2023-24
        if not 2016 <= season_start_year <= 2023:
            raise ValueError("season_start_year must be between 2016 and 2023")
        # Build season code
        season = int(f"{season_start_year}{season_start_year + 1}")
        cache_path = self.data_dir / "_metadata" / "seasons" / f"{season}.json"
        # Use metadata cache if it exists and refresh is not requested
        from_cache = cache_path.exists() and not refresh
        if from_cache:
            metadata = self._load_game(cache_path)
        else:
            # Fetch metadata from the NHL API, using the Cayenne query parameter to filter by season and request all records
            metadata = self._request_json(
                self.GAME_LIST_URL,
                params={"cayenneExp": f"season={season}", "limit": -1},
            )

        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("data"), list)
            or type(metadata.get("total")) is not int
            or len(metadata["data"]) != metadata["total"]
        ):
            raise ValueError(
                "Season game list must contain all records, not a partial page"
            )

        # Set removes duplicates, then sort for consistent order. Validate each record's season and game ID.
        game_ids = set()
        for record in metadata["data"]:
            if not isinstance(record, dict) or record.get("season") != season:
                raise ValueError(
                    "Season metadata contains an invalid/different-season record"
                )
            # Only include completed regular-season and playoff games, excluding unplayed 'if necessary' playoff games
            if record.get("gameType") in (2, 3) and record.get("gameStateId") in (6, 7):
                game_id = record.get("id")
                if (
                    type(game_id) is not int
                    or not 1_000_000_000 <= game_id <= 9_999_999_999
                ):
                    raise ValueError("Season metadata contains an invalid game ID")
                game_ids.add(game_id)

        if not from_cache:
            self._save_game(cache_path, metadata)
        return sorted(game_ids)

    def get_season(
        self,
        season_start_year: int,
        *,
        max_games: int | None = None,
        game_ids: list[int] | None = None,
        refresh: bool = False,
    ) -> dict:
        """Retrieve a season's games, or an explicitly requested subset.

        Args:
            season_start_year: Season's starting year, example: 2023.
            max_games: Optional positive limit. If omitted, use all discovered IDs.
            game_ids: Optional specific discovered IDs, in the desired order.
                If omitted, use sorted IDs; max_games can limit that list.
            refresh: Refresh discovery metadata only, not existing game files.

        Returns:
            Season/discovery count, selected/succeeded IDs, failures with error
            messages, and cached/downloaded counts. Full games stay on disk
            rather than accumulating every payload in memory. The report covers
            only the selected subset, not proof of full-season completeness.
        """
        if max_games is not None and (type(max_games) is not int or max_games < 1):
            raise ValueError("max_games must be a positive integer")
        if game_ids is not None and not isinstance(game_ids, list):
            raise TypeError("game_ids must be a list of integers")

        discovered = self.get_season_game_ids(season_start_year, refresh=refresh)
        if game_ids is None:
            selected = discovered
        else:
            selected = []
            for game_id in game_ids:
                if type(game_id) is not int or game_id not in discovered:
                    raise ValueError(
                        f"Game {game_id!r} is not a completed game in this season"
                    )
                if game_id not in selected:
                    selected.append(game_id)

        if max_games is not None:
            selected = selected[:max_games]

        # Build the report dictionary, then attempt to load or fetch each selected game, recording successes and failures.
        report = {
            "season": int(f"{season_start_year}{season_start_year + 1}"),
            "discovered": len(discovered),
            "selected": selected,
            "succeeded": [],
            "failed": {},
            "cached": 0,
            "downloaded": 0,
        }
        # Attempt to load or fetch each selected game, recording successes and failures.
        for game_id in selected:
            was_cached = self._game_cache_path(game_id).exists()
            try:
                self.get_game(game_id)
            except (requests.RequestException, ValueError, OSError) as exc:
                report["failed"][game_id] = f"{type(exc).__name__}: {exc}"
            else:
                report["succeeded"].append(game_id)
                report["cached" if was_cached else "downloaded"] += 1
        return report

    def _load_game(self, cache_path: Path) -> dict:
        """Read cached JSON, without HTTP requests or pacing waits.
        
        Args:
            cache_path: Path to the existing JSON file. Path contains the game ID in its filename.
            
        Returns:
            A dictionary containing the game's raw JSON data.
        
        Raises:
            ValueError: The cache file is not valid JSON.
        """
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                game = json.load(file)
        except ValueError as exc:
            raise ValueError(f"Invalid cache {cache_path}: {exc}") from exc
        return game
    
    def _fetch_game(self, game_id: int) -> dict:
        """Request and decode one game's JSON using shared pacing/timeout, when not cached.
        
        Args: 
            game_id: Ten digit integer, example: 2023021190
            
        Returns:
            A dictionary containing the game's raw JSON data.
        """
        url = f"{self.BASE_URL}/gamecenter/{game_id}/play-by-play"
        return self._request_json(url)

    def _request_json(self, url: str, *, params: dict | None = None) -> dict:
        """Pace all game/metadata HTTP requests, then check status/decode JSON.
        
        Args: 
            url: Full URL to request.
            params: Optional query parameters for the GET request.
        
        Returns:
            A dictionary containing the JSON data from the response.
        
        Raises:
            requests.RequestException: The HTTP request failed or returned an error status.
            ValueError: The response is not valid JSON.
        """
        # Wait to respect the minimum request interval, if a previous request was made, not for cached data
        if self._last_request_started is not None:
            elapsed = time.monotonic() - self._last_request_started
            wait = self.min_request_interval - elapsed
            if wait > 0:
                time.sleep(wait)
        self._last_request_started = time.monotonic()

        kwargs = {"timeout": self.timeout}
        if params is not None:
            kwargs["params"] = params
        # Send get request
        response = requests.get(url, **kwargs)
        # Turn HTTP errors into exceptions, including 4xx and 5xx status codes
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError(f"Response from {url} is not valid JSON") from exc
        return data
          
    def _save_game(self, cache_path: Path, game: dict) -> None:
        """Write temporary JSON, then atomically move it to the final filename.

        Args:
            cache_path: Path to the JSON file to write. Parent directories are created.
            game: Dictionary containing the game's raw JSON data.
        """
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                json.dump(game, file, ensure_ascii=False, allow_nan=False, indent=2)
                file.write("\n")
            temporary_path.replace(cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)  
                    
def get_player_stats(year: int, player_type: str) -> pd.DataFrame:
    """

    Uses Pandas' built in HTML parser to scrape the tabular player statistics from
    https://www.hockey-reference.com/leagues/ . If the player played on multiple 
    teams in a single season, the individual team's statistics are discarded and
    the total ('TOT') statistics are retained (the multiple team names are discarded)

    Args:
        year (int): The first year of the season to retrieve, i.e. for the 2016-17
            season you'd put in 2016
        player_type (str): Either 'skaters' for forwards and defensemen, or 'goalies'
            for goaltenders.
    """

    if player_type not in ["skaters", "goalies"]:
        raise RuntimeError("'player_type' must be either 'skaters' or 'goalies'")
    
    url = f'https://www.hockey-reference.com/leagues/NHL_{year}_{player_type}.html'

    print(f"Retrieving data from '{url}'...")

    # Use Pandas' built in HTML parser to retrieve the tabular data from the web data
    # Uses BeautifulSoup4 in the background to do the heavylifting
    df = pd.read_html(url, header=1)[0]

    # get players which changed teams during a season
    players_multiple_teams = df[df['Tm'].isin(['TOT'])]

    # filter out players who played on multiple teams
    df = df[~df['Player'].isin(players_multiple_teams['Player'])]
    df = df[df['Player'] != "Player"]

    # add the aggregate rows
    df = pd.concat([df, players_multiple_teams], ignore_index=True)

    return df

def main() -> None:
    """Load the project's .env once, then display configuration only."""
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    retriever = NHLDataRetriever(project_root=project_root)
    print(f"Project root: {retriever.project_root}")
    print(f"Data directory: {retriever.data_dir}")
    print(f"Request timeout: {retriever.timeout} seconds")
    print(f"Minimum request interval: {retriever.min_request_interval} seconds")
    print("Configuration only: no requests, game loading, or directory creation.")


if __name__ == "__main__":
    main()
