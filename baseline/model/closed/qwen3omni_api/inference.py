import base64
import os
import time
from pathlib import Path

import dashscope
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_RETRY_SLEEP = 10

_EXT_TO_MIME = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}


class ClipPayloadTooLargeError(RuntimeError):
    """Local clip cannot be sent under DashScope per-item data-uri / inline payload limits (~20MiB)."""


def _max_data_uri_payload_bytes() -> int:
    return max(4096, int(os.getenv("DASHSCOPE_MAX_DATA_URI_BYTES", "20971520")))


def _base64_len_for_raw(n: int) -> int:
    return 4 * ((int(n) + 2) // 3)


def _estimated_data_uri_byte_len(path: str) -> int:
    p = Path(path).expanduser().resolve()
    raw = p.stat().st_size
    ext = p.suffix.lower()
    mime = _EXT_TO_MIME.get(ext) or ("video/mp4" if ext == ".mov" else "application/octet-stream")
    prefix = len("data:") + len(mime) + len(";base64,")
    return prefix + _base64_len_for_raw(raw)


def _file_exceeds_data_uri_limit(path: str) -> bool:
    try:
        return _estimated_data_uri_byte_len(path) > _max_data_uri_payload_bytes()
    except OSError:
        return True


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


def _openai_split_av_requested(media_type: str, split_visual: str | None, split_audio: str | None) -> bool:
    if media_type != "both" or not _env_truthy("DASHSCOPE_OPENAI_AV_SPLIT"):
        return False
    if not split_visual or not split_audio:
        return False
    return Path(split_visual).is_file() and Path(split_audio).is_file()


# Beijing (default SDK) vs international endpoint; override with full URL if needed.
_DEFAULT_BASE_URLS = {
    "intl": "https://dashscope-intl.aliyuncs.com/api/v1",
    "cn": "https://dashscope.aliyuncs.com/api/v1",
}


def _configure_dashscope_base_url():
    explicit = os.getenv("DASHSCOPE_BASE_HTTP_API_URL")
    if explicit:
        dashscope.base_http_api_url = explicit.strip().rstrip("/")
        return
    region = (os.getenv("DASHSCOPE_REGION") or "intl").lower()
    dashscope.base_http_api_url = _DEFAULT_BASE_URLS.get(region, _DEFAULT_BASE_URLS["intl"])


def _local_media_uri(path: str) -> str:
    return Path(path).resolve().as_uri()


def _ensure_clip_exists(media_path: str, *, hint: str = "") -> None:
    """Fail fast without API retries when zips/chunks were not fully extracted."""
    p = Path(media_path).expanduser()
    if not p.is_file():
        msg = (
            "Media clip not found locally — unzip/download all chunks under `--data_path` "
            f"so annotations match files on disk. Missing: {p.resolve()}"
        )
        if hint:
            msg = f"{msg} ({hint})"
        raise FileNotFoundError(msg)


def _normalize_answer_text(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def _local_file_data_url(media_path: str) -> str:
    p = Path(media_path).expanduser().resolve()
    ext = p.suffix.lower()
    mime = _EXT_TO_MIME.get(ext) or ("video/mp4" if ext == ".mov" else "application/octet-stream")
    b64 = base64.standard_b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _looks_like_media_url_rejected(exc: BaseException) -> bool:
    s = str(exc).lower()
    if "the provided url" in s:
        return True
    compact = s.replace(" ", "")
    if "invalidparameter" in compact and "url" in s:
        return True
    return "url" in s and ("invalid" in s or "not appear" in s or "formatted" in s)


def _is_openai_data_uri_item_too_large(exc: BaseException) -> bool:
    """DashScope OpenAI compat caps each ``data:...;base64,...`` item (~20MiB, error text may say 20971520)."""
    s = str(exc).lower()
    if "max bytes per data-uri" in s:
        return True
    if "data-uri" in s and "exceeded limit" in s:
        return True
    if "20971520" in s and ("data" in s or "uri" in s):
        return True
    if "badrequest.toolarge" in s.replace(" ", ""):
        return True
    if "too large" in s and ("data-uri" in s or "data uri" in s):
        return True
    return False


def _native_multimodal_once(
    api_model_id: str,
    media_type: str,
    media_path: str,
    question: str,
    fps: float,
    *,
    allow_inline_base64: bool = True,
) -> str:
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()

    prefer_data = _env_truthy("DASHSCOPE_NATIVE_PREFER_DATA_URL")
    allow_file_uri = os.getenv("DASHSCOPE_NATIVE_ALLOW_FILE_URI", "1").strip().lower() in ("1", "true", "yes")

    fits_embed = allow_inline_base64 and not _file_exceeds_data_uri_limit(media_path)

    if media_type == "audio":
        if prefer_data:
            if not fits_embed:
                raise ClipPayloadTooLargeError(
                    f"Audio clip too large for DashScope inline/data-uri limit: {media_path}"
                )
            refs: list[tuple[str, str]] = [(_local_file_data_url(media_path), "data-url")]
        elif allow_file_uri:
            refs = [(_local_media_uri(media_path), "file-uri")]
            if fits_embed:
                refs.append((_local_file_data_url(media_path), "data-url"))
        else:
            if not fits_embed:
                raise ClipPayloadTooLargeError(
                    f"Audio clip too large for DashScope inline/data-uri limit: {media_path}"
                )
            refs = [(_local_file_data_url(media_path), "data-url")]
    elif prefer_data:
        if not fits_embed:
            raise ClipPayloadTooLargeError(
                f"Video clip too large for DashScope inline/data-uri limit: {media_path}"
            )
        refs = [(_local_file_data_url(media_path), "data-url")]
    elif allow_file_uri:
        refs = [(_local_media_uri(media_path), "file-uri")]
        if fits_embed:
            refs.append((_local_file_data_url(media_path), "data-url"))
    else:
        if not fits_embed:
            raise ClipPayloadTooLargeError(
                f"Video clip too large for DashScope inline/data-uri limit: {media_path}"
            )
        refs = [(_local_file_data_url(media_path), "data-url")]

    if not refs:
        raise ClipPayloadTooLargeError(f"No usable native transport for clip: {media_path}")

    last_err: RuntimeError | None = None
    for ref, tag in refs:
        content: list = []
        if media_type == "audio":
            content.append({"audio": ref})
        else:
            content.append({"video": ref, "fps": fps})
        content.append({"text": question})
        messages = [{"role": "user", "content": content}]
        try:
            response = dashscope.MultiModalConversation.call(
                api_key=api_key,
                model=api_model_id,
                messages=messages,
                incremental_output=False,
                stream=False,
                enable_thinking=False,
            )
            if response.status_code != 200:
                msg = f"{getattr(response, 'code', '')} {getattr(response, 'message', '')}"
                if _is_openai_data_uri_item_too_large(RuntimeError(msg)):
                    raise ClipPayloadTooLargeError(
                        f"DashScope rejected inline payload (too large): {media_path} ({msg})"
                    )
                raise RuntimeError(f"DashScope HTTP {response.status_code}: {msg}")
            parts = response.output.choices[0].message.content
            text = parts[0].get("text", "") if parts else ""
            return _normalize_answer_text(text)
        except ClipPayloadTooLargeError:
            raise
        except RuntimeError as e:
            last_err = e
            if _is_openai_data_uri_item_too_large(e):
                raise ClipPayloadTooLargeError(str(e)) from e
            if tag == "file-uri" and len(refs) > 1 and _looks_like_media_url_rejected(e):
                print(
                    "DashScope rejected local file URI; retrying same clip as inline data URL (slower, larger payload)."
                )
                continue
            if (
                tag == "file-uri"
                and len(refs) == 1
                and (not fits_embed or not allow_inline_base64)
                and _looks_like_media_url_rejected(e)
            ):
                raise ClipPayloadTooLargeError(
                    f"DashScope rejected file:// and clip cannot be inlined under size cap: {media_path}"
                ) from e
            raise

    raise last_err if last_err else RuntimeError("DashScope native multimodal produced no response")


def _import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "Set DASHSCOPE_OPENAI_BASE_URL requires the OpenAI SDK: pip install 'openai>=1.52.0'"
        ) from e
    return OpenAI


def _openai_compatible_stream_answer(client, api_model_id: str, mm_content: list) -> str:
    """
    Qwen-Omni on DashScope-compatible OpenAI endpoint requires ``stream=True`` (see阿里云文档).
    We only aggregate **text** (``modalities=[\"text\"]``) — no synthesized speech output / TTS.
    """
    stream = client.chat.completions.create(
        model=api_model_id,
        messages=[{"role": "user", "content": mm_content}],
        stream=True,
        stream_options={"include_usage": True},
        modalities=["text"],
    )
    pieces: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta
        if delta is None:
            continue
        part = getattr(delta, "content", None)
        if part:
            pieces.append(part)
    return "".join(pieces)


def _openai_compatible_once(api_model_id: str, media_type: str, media_path: str, question: str, fps: float) -> str:
    """Official OpenAI-compatible ``/v1/chat/completions`` (local media as ``data:*;base64``)."""
    _ensure_clip_exists(media_path, hint="combined / single modality clip")
    if _file_exceeds_data_uri_limit(media_path):
        raise ClipPayloadTooLargeError(
            f"Clip exceeds DashScope data-uri byte limit ({_max_data_uri_payload_bytes()}): {media_path}"
        )
    OpenAI = _import_openai_client()

    base_url = (os.getenv("DASHSCOPE_OPENAI_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()

    client = OpenAI(api_key=api_key, base_url=base_url)
    url = _local_file_data_url(media_path)
    mm_content: list[dict] = [
        {"type": "video_url", "video_url": {"url": url}, "fps": fps},
        {"type": "text", "text": question},
    ]
    ans = _openai_compatible_stream_answer(client, api_model_id, mm_content)
    return _normalize_answer_text(ans)


def _openai_compatible_split_av_once(
    api_model_id: str, visual_path: str, audio_path: str, question: str, fps: float
) -> str:
    """
    Multi-part ``content`` (like docs: ``image_url`` + ``input_audio`` + text).

    Docs pass **HTTPS URLs** in ``input_audio.data``; for local WAV we send a **full data URL**
    (``data:audio/wav;base64,...``) so it still parses as a URL string.

    Note: ``video_url`` + ``input_audio`` is not spelled out as an example everywhere; callers
    wrap with a fallback to a single **audiovisual** MP4 on failure.
    """
    _ensure_clip_exists(visual_path, hint="visual_only")
    _ensure_clip_exists(audio_path, hint="audio_only")
    for p in (visual_path, audio_path):
        if _file_exceeds_data_uri_limit(p):
            raise ClipPayloadTooLargeError(
                f"Clip exceeds DashScope data-uri byte limit ({_max_data_uri_payload_bytes()}): {p}"
            )
    OpenAI = _import_openai_client()

    base_url = (os.getenv("DASHSCOPE_OPENAI_BASE_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    client = OpenAI(api_key=api_key, base_url=base_url)

    v_url = _local_file_data_url(visual_path)
    audio_data_url = _local_file_data_url(audio_path)

    mm_content: list[dict] = [
        {"type": "video_url", "video_url": {"url": v_url}, "fps": fps},
        {"type": "input_audio", "input_audio": {"data": audio_data_url, "format": "wav"}},
        {"type": "text", "text": question},
    ]
    ans = _openai_compatible_stream_answer(client, api_model_id, mm_content)
    return _normalize_answer_text(ans)


def qwen3omni_api_process(
    api_model_id,
    media_type,
    media_path,
    question,
    split_visual_path: str | None = None,
    split_audio_path: str | None = None,
):
    """
    Call ``qwen3.5-omni-plus-2026-03-15`` (or another DashScope omni id) on local clips.

    * **Native (default):** ``dashscope.MultiModalConversation``. Tries ``file://`` first;
      if the gateway rejects it (common on CN: "The provided URL does not appear …"),
      retries **once** with the same clip as ``data:*;base64,...``. Optionally set
      ``DASHSCOPE_NATIVE_PREFER_DATA_URL=1`` to skip ``file://`` entirely, or
      ``DASHSCOPE_NATIVE_ALLOW_FILE_URI=0`` to always inline base64.

    * **Optional — Model Studio OpenAI examples:** set
      ``DASHSCOPE_OPENAI_BASE_URL`` to the compatible endpoint, e.g. China
      ``https://dashscope.aliyuncs.com/compatible-mode/v1`` or international
      ``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``. Then video /
      audiovisual requests use ``chat.completions`` with a ``video_url`` payload
      (local file as ``data:...;base64,...``), unless ``DASHSCOPE_OPENAI_AV_SPLIT=1``:
      same pattern as docs (multiple content parts), using **``video_url`` + ``input_audio``**
      + text from ``visual_only`` / ``audio_only`` paths. **Audio-only** still uses the
      native multimodal API (SDK file URI), which is more reliable for ``.wav``.

    Benchmark note: Qwen-Omni on DashScope requires ``stream=True``; we aggregate text only via
      ``modalities=["text"]`` (no synthesized speech ``audio={...}`` in the response).

    Environment:
      DASHSCOPE_API_KEY — required
      DASHSCOPE_REGION — optional for native path, ``intl`` (default) or ``cn``
      DASHSCOPE_BASE_HTTP_API_URL — optional; overrides region for native SDK
      DASHSCOPE_NATIVE_PREFER_DATA_URL — optional; ``1`` = only inline base64 for native path
      DASHSCOPE_NATIVE_ALLOW_FILE_URI — optional; ``0`` = skip ``file://``, use base64 only
      DASHSCOPE_OPENAI_BASE_URL — optional; enable OpenAI-compatible transport
      DASHSCOPE_OPENAI_AV_SPLIT — optional; ``1`` = ``video_url`` + ``input_audio`` (split clips) instead of one combined mp4
      QWEN3_OMNI_API_FPS — optional video fps (default 2.0)
      DASHSCOPE_MAX_RETRIES — optional; transient API retries (default ``5``)
      DASHSCOPE_RETRY_SLEEP_SEC — optional; sleep between retries seconds (default ``10``, was historically 40)
      DASHSCOPE_MAX_DATA_URI_BYTES — optional; per-item data-uri cap used for pre-checks (default ``20971520``)
    """
    api_key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY for DashScope Qwen3-Omni API inference.")

    fps = float(os.getenv("QWEN3_OMNI_API_FPS", "2"))
    openai_base = (os.getenv("DASHSCOPE_OPENAI_BASE_URL") or "").strip()
    use_openai = bool(openai_base) and media_type != "audio"

    if not use_openai:
        _configure_dashscope_base_url()

    if use_openai and _openai_split_av_requested(media_type, split_visual_path, split_audio_path):
        print(
            media_type,
            split_visual_path,
            split_audio_path,
            "(openai-compatible split: video_url + input_audio)",
        )
    else:
        print(media_type, media_path, _local_media_uri(media_path) if not use_openai else "(openai-compatible)")

    _split_take = use_openai and _openai_split_av_requested(media_type, split_visual_path, split_audio_path)
    if _split_take:
        _ensure_clip_exists(split_visual_path, hint="visual_only")
        _ensure_clip_exists(split_audio_path, hint="audio_only")
    else:
        _ensure_clip_exists(media_path)

    max_retries = max(1, int(os.getenv("DASHSCOPE_MAX_RETRIES", "5")))
    retry_sleep = float(os.getenv("DASHSCOPE_RETRY_SLEEP_SEC", str(_DEFAULT_RETRY_SLEEP)))
    if retry_sleep < 0:
        retry_sleep = 0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if use_openai:
                if _openai_split_av_requested(media_type, split_visual_path, split_audio_path):
                    try:
                        return _openai_compatible_split_av_once(
                            api_model_id, split_visual_path, split_audio_path, question, fps
                        )
                    except ClipPayloadTooLargeError:
                        raise
                    except FileNotFoundError:
                        raise
                    except Exception as e:
                        print(
                            "OpenAI split (video_url + input_audio) failed "
                            f"({type(e).__name__}: {e}); falling back to single audiovisual MP4."
                        )
                        return _openai_compatible_once(
                            api_model_id, media_type, media_path, question, fps
                        )
                return _openai_compatible_once(api_model_id, media_type, media_path, question, fps)
            return _native_multimodal_once(
                api_model_id, media_type, media_path, question, fps, allow_inline_base64=True
            )
        except ClipPayloadTooLargeError:
            if use_openai:
                print(
                    "Clip too large for OpenAI data-uri (or native inline); "
                    "retrying native DashScope with file:// only (no base64 fallback)."
                )
                _configure_dashscope_base_url()
                return _native_multimodal_once(
                    api_model_id, media_type, media_path, question, fps, allow_inline_base64=False
                )
            raise
        except FileNotFoundError:
            raise
        except Exception as e:
            if use_openai and _is_openai_data_uri_item_too_large(e):
                print(
                    "OpenAI compatible: data-uri item exceeds gateway size limit; "
                    "using native DashScope with file:// only (no base64 fallback)."
                )
                _configure_dashscope_base_url()
                try:
                    return _native_multimodal_once(
                        api_model_id, media_type, media_path, question, fps, allow_inline_base64=False
                    )
                except ClipPayloadTooLargeError:
                    raise
            last_err = e
            print(
                f"DashScope Qwen3-Omni API exception (attempt {attempt + 1}/{max_retries}): {e}; "
                f"retry in {retry_sleep}s."
            )
            time.sleep(retry_sleep)
    raise RuntimeError(f"DashScope Qwen3-Omni API failed after {max_retries} attempts") from last_err
