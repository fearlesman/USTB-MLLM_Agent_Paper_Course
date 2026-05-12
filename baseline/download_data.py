"""Download a Hugging Face dataset snapshot (mirror-friendly, default: Holistic_AVQA_bench).

`hf download` talks to `https://huggingface.co` for recursive tree listing; on flaky routes
that can raise httpx.ConnectTimeout / WinError 10060 while blob transfers still progress.
This script defaults to HF_ENDPOINT=https://hf-mirror.com and retries snapshot + verify.

By default this script disables TLS certificate verification for Hub requests (insecure;
traffic can be intercepted). Set HF_HUB_VERIFY_SSL=1 to enable verification and normal
CA handling (optionally with certifi / SSL_CERT_FILE).

Mirror: default hf-mirror.com. Official hub: HF_ENDPOINT=https://huggingface.co
Other repo: HF_SNAPSHOT_REPO_ID=namespace/name (or HOLISTIC_AVQA_REPO_ID).

Default local snapshot folder: ``<repo>/Holistic_AVQA_bench`` (same as
``hf download ... --local-dir ./Holistic_AVQA_bench`` from repo root).
Override with HOLISTIC_AVQA_LOCAL_DIR (e.g. a separate path for AV-SpeakerBench).

If verify still reports a small file gap after skips, set HF_SNAPSHOT_VERIFY_MAX_SHORT=N
to accept at most N missing files (optional), or HF_SKIP_SNAPSHOT_VERIFY=1 to skip verify.

Mirrors may return paginated ``Link: rel="next"`` headers with URLs still pointing at huggingface.co;
this script rewrites those to the current ``HF_ENDPOINT`` so tree listing stays on the mirror.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

# Mirror: default hf-mirror.com. Official hub: HF_ENDPOINT=https://huggingface.co
HUB_ENDPOINT = (
    os.environ.get("HF_ENDPOINT")
    or os.environ.get("HF_MIRROR")
    or "https://hf-mirror.com"
).rstrip("/")
os.environ["HF_ENDPOINT"] = HUB_ENDPOINT

# Longer Hub API timeouts before importing huggingface_hub (library reads env at import).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


def _hub_verify_ssl_enabled() -> bool:
    return os.environ.get("HF_HUB_VERIFY_SSL", "0").lower() in ("1", "true", "yes")


def _use_certifi_ca_bundle() -> None:
    """When verification is on: help Windows/Anaconda find a current CA bundle."""
    if not _hub_verify_ssl_enabled():
        return
    if os.environ.get("HF_USE_SYSTEM_CA", "").lower() in ("1", "true", "yes"):
        return
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    os.environ["SSL_CERT_FILE"] = bundle
    os.environ["REQUESTS_CA_BUNDLE"] = bundle


_use_certifi_ca_bundle()

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


def _apply_hub_http_backend() -> None:
    if _hub_verify_ssl_enabled():
        return
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        import httpx
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import hf_request_event_hook

        def factory() -> httpx.Client:
            return httpx.Client(
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
                verify=False,
            )

        set_client_factory(factory)
        return
    except ImportError:
        pass

    import requests
    from huggingface_hub import configure_http_backend

    def factory_rs() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=factory_rs)


def _reset_hub_sessions() -> None:
    try:
        from huggingface_hub.utils import reset_sessions

        reset_sessions()
    except ImportError:
        from huggingface_hub import close_session

        close_session()
    except Exception:
        try:
            from huggingface_hub import close_session

            close_session()
        except Exception:
            pass


_apply_hub_http_backend()


def _install_hf_pagination_host_rewrite() -> None:
    """Keep repo tree pagination on HF_ENDPOINT (mirrors often advertise next=https://huggingface.co/...)."""
    import huggingface_hub.utils._pagination as pag
    import huggingface_hub.hf_api as hf_api
    import huggingface_hub.utils as hf_utils
    from huggingface_hub.utils import get_session, hf_raise_for_status, http_backoff
    from huggingface_hub.utils import logging as hf_logging

    logger = hf_logging.get_logger(__name__)
    _get_next = pag._get_next_page

    def _rewrite(url: str | None) -> str | None:
        if not url:
            return url
        preferred = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        for prefix in ("https://huggingface.co", "http://huggingface.co"):
            if url.startswith(prefix):
                return preferred + url[len(prefix) :]
        return url

    def paginate(path: str, params: dict, headers: dict):
        session = get_session()
        path = _rewrite(path) or path
        r = session.get(path, params=params, headers=headers)
        hf_raise_for_status(r)
        yield from r.json()

        next_page = _rewrite(_get_next(r))
        while next_page is not None:
            logger.debug("Pagination detected. Requesting next page: %s", next_page)
            r = http_backoff("GET", next_page, headers=headers)
            hf_raise_for_status(r)
            yield from r.json()
            next_page = _rewrite(_get_next(r))

    # Important: hf_api imports `paginate` at module import time. Patch all call sites.
    pag.paginate = paginate
    hf_utils.paginate = paginate
    hf_api.paginate = paginate


_install_hf_pagination_host_rewrite()

try:
    import requests
except ImportError:
    requests = None  # type: ignore
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

_BASELINE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASELINE_ROOT.parent
# Default matches: hf download ... --local-dir ./Holistic_AVQA_bench (next to ``baseline/``, not inside it).
local_dir = os.environ.get("HOLISTIC_AVQA_LOCAL_DIR", str(_PROJECT_ROOT / "Holistic_AVQA_bench"))

OFFICIAL_HUB = "https://huggingface.co"
REPO_ID = (
    os.environ.get("HF_SNAPSHOT_REPO_ID")
    or os.environ.get("HOLISTIC_AVQA_REPO_ID")
    or "plnguyen2908/Holistic_AVQA_bench"
)
# Pin a git revision to avoid flaky /api/.../revision/main resolution (optional).
_SNAPSHOT_REVISION = os.environ.get("HOLISTIC_AVQA_REVISION") or None

# macOS junk in repo; hf-mirror often 403s on .DS_Store — skip so download can finish.
_IGNORE = [".DS_Store", "**/.DS_Store"]
# Runtime-added ignore list for files that are permanently missing on current endpoint (404).
_DYNAMIC_IGNORE: list[str] = []


def _norm_hub_relpath(p: str) -> str:
    """Normalize repo-relative paths for Hub (always POSIX-style)."""
    return p.replace("\\", "/").strip().lstrip("./")


def _ignore_patterns() -> list[str]:
    return [*_IGNORE, *_DYNAMIC_IGNORE]


def _register_dynamic_ignore(path: str) -> None:
    n = _norm_hub_relpath(path)
    if n and n not in _DYNAMIC_IGNORE:
        _DYNAMIC_IGNORE.append(n)


def _endpoint_chain(primary: str) -> list[str]:
    """Try primary first (often hf-mirror); mirror intermittently 403s real files — then official hub."""
    primary = primary.rstrip("/")
    out: list[str] = [primary]
    if os.environ.get("HF_NO_OFFICIAL_FALLBACK", "").lower() in ("1", "true", "yes"):
        return out
    if primary != OFFICIAL_HUB.rstrip("/"):
        out.append(OFFICIAL_HUB.rstrip("/"))
    return out


def _run_snapshot(endpoint: str, *, max_workers: int, etag_timeout: float) -> None:
    kwargs = dict(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        endpoint=endpoint,
        ignore_patterns=_ignore_patterns(),
        max_workers=max_workers,
        etag_timeout=etag_timeout,
    )
    if _SNAPSHOT_REVISION:
        kwargs["revision"] = _SNAPSHOT_REVISION
    snapshot_download(**kwargs)


def _count_local_snapshot_files(root: Path) -> int:
    """Count files under snapshot root, excluding Hub metadata cache."""
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".cache" in p.relative_to(root).parts:
            continue
        n += 1
    return n


def _local_snapshot_relpaths(root: Path) -> set[str]:
    """Repo-relative POSIX paths of all regular files under snapshot (excluding Hub cache)."""
    out: set[str] = set()
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".cache" in p.relative_to(root).parts:
            continue
        out.add(_norm_hub_relpath(str(p.relative_to(root))))
    return out


def _verify_snapshot_complete(endpoint: str) -> None:
    """snapshot_download may return early with partial data when Hub returns 502 but local_dir is non-empty."""
    if os.environ.get("HF_SKIP_SNAPSHOT_VERIFY", "").lower() in ("1", "true", "yes"):
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    api = HfApi(endpoint=endpoint)
    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type="dataset", revision=_SNAPSHOT_REVISION)
    except Exception as e:
        raise LocalEntryNotFoundError(
            "snapshot_download returned but Hub is unreachable for verification (e.g. 502). "
            "Local data may be incomplete — will retry."
        ) from e

    sibs = info.siblings or []
    # Static ignores only, then drop 403/404 skips by exact path (fnmatch on dynamic paths is unreliable).
    skip_set = {_norm_hub_relpath(p) for p in _DYNAMIC_IGNORE}
    filtered = list(filter_repo_objects(sibs, ignore_patterns=_IGNORE, key=lambda s: s.rfilename))
    filtered = [s for s in filtered if _norm_hub_relpath(s.rfilename) not in skip_set]
    expected = len(filtered)
    if expected == 0:
        raise LocalEntryNotFoundError("Hub returned no siblings; cannot verify snapshot.")

    root = Path(local_dir)
    got = _count_local_snapshot_files(root)
    max_short_raw = os.environ.get("HF_SNAPSHOT_VERIFY_MAX_SHORT", "0")
    try:
        max_short = max(0, int(max_short_raw))
    except ValueError:
        max_short = 0

    if got >= expected:
        print(f"Verified: {got} local files >= {expected} on Hub.")
        return

    short_by = expected - got
    hub_paths = {_norm_hub_relpath(s.rfilename) for s in filtered}
    local_paths = _local_snapshot_relpaths(root)
    missing_sorted = sorted(hub_paths - local_paths)
    sample = missing_sorted[:20]

    if max_short > 0 and short_by <= max_short:
        print(
            f"Verify relaxed (HF_SNAPSHOT_VERIFY_MAX_SHORT={max_short}): "
            f"{got} local vs {expected} expected, short by {short_by}. "
            f"Treating as OK. Missing sample: {sample}"
        )
        return

    raise LocalEntryNotFoundError(
        f"Incomplete snapshot: {got} files under {local_dir} (excluding .cache) vs {expected} on Hub "
        f"(after static ignore and {len(skip_set)} dynamic skip path(s)). "
        f"Missing ~{short_by} file(s); sample: {sample}. "
        f"Set HF_SNAPSHOT_VERIFY_MAX_SHORT={short_by} to allow this gap, or HF_SKIP_SNAPSHOT_VERIFY=1 to skip check."
    )


def _is_http_403(exc: BaseException) -> bool:
    """True if this or a chained HfHubHTTPError is HTTP 403 (mirror often blocks specific blobs permanently)."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HfHubHTTPError):
            resp = getattr(cur, "response", None)
            code = getattr(resp, "status_code", None) if resp is not None else None
            if code == 403:
                return True
        cur = cur.__cause__ or getattr(cur, "__context__", None)
    msg = str(exc)
    return "403" in msg and "Forbidden" in msg


def _is_http_404(exc: BaseException) -> bool:
    """True if this or a chained HfHubHTTPError is HTTP 404."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HfHubHTTPError):
            resp = getattr(cur, "response", None)
            code = getattr(resp, "status_code", None) if resp is not None else None
            if code == 404:
                return True
        cur = cur.__cause__ or getattr(cur, "__context__", None)
    msg = str(exc)
    return "404" in msg and ("Not Found" in msg or "Entry Not Found" in msg)


def _extract_missing_path_from_404(exc: BaseException) -> str | None:
    """Best-effort extraction of missing repo path from 404 error URL."""
    url: str | None = None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HfHubHTTPError):
            resp = getattr(cur, "response", None)
            if resp is not None:
                req = getattr(resp, "request", None)
                if req is not None and getattr(req, "url", None) is not None:
                    url = str(req.url)
                    break
                if getattr(resp, "url", None) is not None:
                    url = str(resp.url)
                    break
        cur = cur.__cause__ or getattr(cur, "__context__", None)

    if not url:
        m = re.search(r"Entry Not Found for url:\s*(https?://\S+)", str(exc))
        if m:
            url = m.group(1)
    if not url:
        return None

    parsed = urlparse(url)
    # /datasets/{org}/{repo}/resolve/{revision}/{path...}
    m = re.search(r"/datasets/[^/]+/[^/]+/resolve/[^/]+/(.+)$", parsed.path)
    if not m:
        return None
    return unquote(m.group(1))


def _extract_missing_path_from_hub_error(exc: BaseException) -> str | None:
    """Best-effort extraction of missing repo path from 403/404 error URL."""
    url: str | None = None
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HfHubHTTPError):
            resp = getattr(cur, "response", None)
            if resp is not None:
                req = getattr(resp, "request", None)
                if req is not None and getattr(req, "url", None) is not None:
                    url = str(req.url)
                    break
                if getattr(resp, "url", None) is not None:
                    url = str(resp.url)
                    break
        cur = cur.__cause__ or getattr(cur, "__context__", None)

    if not url:
        patterns = (
            r"Entry Not Found for url:\s*(https?://\S+)",
            r"Cannot access content at:\s*(https?://\S+)",
        )
        text = str(exc)
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                url = m.group(1)
                break
    if not url:
        return None

    parsed = urlparse(url)
    m = re.search(r"/datasets/[^/]+/[^/]+/resolve/[^/]+/(.+)$", parsed.path)
    if not m:
        return None
    return unquote(m.group(1))


def _is_transient_network_error(exc: BaseException) -> bool:
    """Best-effort detection of retryable transport errors (e.g. RemoteProtocolError)."""
    transient_names = {
        "RemoteProtocolError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "ConnectError",
        "ReadError",
        "WriteError",
        "NetworkError",
        "ProtocolError",
    }
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if httpx is not None and isinstance(cur, (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError)):
            return True
        if type(cur).__name__ in transient_names:
            return True
        cur = cur.__cause__ or getattr(cur, "__context__", None)
    msg = str(exc)
    return "Server disconnected without sending a response" in msg


def main() -> None:
    max_workers = int(os.environ.get("HF_HUB_DOWNLOAD_MAX_WORKERS", "2"))
    etag_timeout = float(os.environ.get("HF_HUB_ETAG_TIMEOUT", "60"))
    retries = int(os.environ.get("HF_SNAPSHOT_RETRIES", "10"))
    base_wait = float(os.environ.get("HF_SNAPSHOT_RETRY_WAIT_SEC", "8"))
    print(f"repo_id={REPO_ID}")
    print(f"local_dir={local_dir}")
    print(f"HF_HUB_VERIFY_SSL={'1' if _hub_verify_ssl_enabled() else '0 (cert check disabled)'}")
    print(
        f"endpoint chain={_endpoint_chain(HUB_ENDPOINT)}  max_workers={max_workers}  "
        f"etag_timeout={etag_timeout}s  retries/endpoint={retries}"
    )
    if os.environ.get("HF_SKIP_404_FILES", "1").lower() in ("1", "true", "yes"):
        print("HF_SKIP_404_FILES=1 (skip permanently missing files on 404)")
    if os.environ.get("HF_SKIP_403_FILES", "1").lower() in ("1", "true", "yes"):
        print("HF_SKIP_403_FILES=1 (skip blocked files on 403 when path is known)")

    last_exc: BaseException | None = None
    endpoints = _endpoint_chain(HUB_ENDPOINT)
    for idx, ep in enumerate(endpoints):
        os.environ["HF_ENDPOINT"] = ep
        print(f"--- trying {ep} ---")
        for attempt in range(1, retries + 1):
            try:
                _reset_hub_sessions()
                _run_snapshot(ep, max_workers=max_workers, etag_timeout=etag_timeout)
                _verify_snapshot_complete(ep)
                print("Done.")
                return
            except (HfHubHTTPError, LocalEntryNotFoundError) as e:
                last_exc = e
                cause = getattr(e, "__cause__", None) or getattr(e, "response", None)
                extra = f"  cause={cause!r}" if cause else ""
                print(f"Attempt {attempt}/{retries} failed: {e!s}{extra}")
                skip_404 = os.environ.get("HF_SKIP_404_FILES", "1").lower() in ("1", "true", "yes")
                skip_403 = os.environ.get("HF_SKIP_403_FILES", "1").lower() in ("1", "true", "yes")
                # 404/403 改为立即重试，不按 attempt 指数加大退避
                no_backoff_wait = False
                if _is_http_404(e) and skip_404:
                    no_backoff_wait = True
                    missing = _extract_missing_path_from_hub_error(e) or _extract_missing_path_from_404(e)
                    if missing:
                        nmiss = _norm_hub_relpath(missing)
                        before = len(_DYNAMIC_IGNORE)
                        _register_dynamic_ignore(missing)
                        if len(_DYNAMIC_IGNORE) > before:
                            print(f"HTTP 404 missing file; will skip: {nmiss}")
                        else:
                            print(f"HTTP 404 still missing (already skipped): {nmiss}")
                    else:
                        print("HTTP 404 detected but could not parse missing path; will retry.")
                if _is_http_403(e):
                    missing403 = _extract_missing_path_from_hub_error(e) if skip_403 else None
                    if missing403:
                        no_backoff_wait = True
                        n403 = _norm_hub_relpath(missing403)
                        before = len(_DYNAMIC_IGNORE)
                        _register_dynamic_ignore(missing403)
                        if len(_DYNAMIC_IGNORE) > before:
                            print(f"HTTP 403 blocked file; will skip: {n403}")
                        else:
                            print(f"HTTP 403 still blocked (already skipped): {n403}")
                    elif idx < len(endpoints) - 1:
                        print("HTTP 403 on this host — mirror often blocks individual files; switching to next endpoint.")
                        break
                if attempt < retries:
                    if no_backoff_wait:
                        print("Retrying immediately (no backoff after 404/403)…")
                    else:
                        wait = min(base_wait * (1.4 ** (attempt - 1)), 180)
                        print(f"Retrying in {wait:.0f}s…")
                        time.sleep(wait)
                else:
                    print("Giving up on this endpoint; switch if another is configured.")
            except OSError as e:
                last_exc = e
                if requests is not None and isinstance(e, requests.exceptions.SSLError):
                    print(f"Failed (SSL): {e!s}")
                    print(
                        "If you meant to skip certificate checks, leave HF_HUB_VERIFY_SSL unset. "
                        "Otherwise: pip install -U certifi, set SSL_CERT_FILE, or HF_USE_SYSTEM_CA=1."
                    )
                else:
                    print(f"Failed (OS): {e!s}")
                break
            except Exception as e:
                last_exc = e
                if _is_transient_network_error(e):
                    print(f"Attempt {attempt}/{retries} transient network error: {e!s}")
                    if attempt < retries:
                        wait = min(base_wait * (1.4 ** (attempt - 1)), 180)
                        print(f"Retrying in {wait:.0f}s…")
                        time.sleep(wait)
                        continue
                raise

    if last_exc:
        raise last_exc
    raise RuntimeError("download_data: no endpoint tried")


if __name__ == "__main__":
    main()
