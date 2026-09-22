"""First-run configuration: what it refuses, and what it must never lose.

This is the only screen a new user is guaranteed to meet, and it writes the file the
rest of the app reads. Two things matter more than the rest:

* a setting the user tuned by hand must survive a trip through the form, because the
  form owns five keys and the file holds sixty;
* a stored API key must survive a save where the key field was left blank, or every
  change of folder silently wipes the key and the app stops answering.

Invoked by ``python rag.py test``.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = "  pass  ", "  FAIL  "
_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print((PASS if ok else FAIL) + name + (f"   {detail}" if detail else ""))
    if not ok:
        _failures.append(name)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="setuptest_"))
    os.environ["RAG_CONFIG_DIR"] = str(tmp)

    from app import setup

    try:
        print("-" * 64)
        print("the settings file goes somewhere the user can write")
        print("-" * 64)
        check("it honours an explicit config directory",
              setup.config_dir() == tmp, str(setup.config_dir()))
        check("...and is not inside the program's own folder",
              ROOT not in setup.env_path().parents, str(setup.env_path()))

        print("\n" + "-" * 64)
        print("what it refuses, and why")
        print("-" * 64)
        cases = [
            ({}, True, "nothing filled in"),
            ({"DATA_DIR": "Z:/definitely/not/here"}, True, "a folder that is not there"),
            ({"DATA_DIR": str(tmp)}, True, "a real folder but no key at all"),
            ({"DATA_DIR": str(tmp), "GROQ_API_KEY": "gsk_x"}, False, "folder plus Groq"),
            ({"DATA_DIR": str(tmp), "GEMINI_API_KEY": "AIza_x"}, False,
             "folder plus Gemini, on its own"),
            ({"DATA_DIR": str(tmp), "GROQ_API_KEY": "k",
              "STORAGE_DIR": "Z:/cannot/write"}, True, "an index folder it cannot write"),
        ]
        for values, should_refuse, why in cases:
            problems = setup.validate(values)
            check(f"{'refuses' if should_refuse else 'accepts'}: {why}",
                  bool(problems) == should_refuse,
                  problems[0][:46] if problems else "")

        # A file is only a problem if the user points at one.
        f = tmp / "a-file.txt"
        f.write_text("x", encoding="utf-8")
        check("refuses: a file where a folder was wanted",
              bool(setup.validate({"DATA_DIR": str(f), "GROQ_API_KEY": "k"})))

        print("\n" + "-" * 64)
        print("saving keeps everything it does not own")
        print("-" * 64)
        path = setup.write_env({"DATA_DIR": str(tmp), "STORAGE_DIR": str(tmp / "index"),
                                "GROQ_API_KEY": "gsk_first", "GEMINI_API_KEY": "",
                                "LLM_MODEL": "groq::x"})
        check("the file is written", path.exists(), str(path))
        back = setup.read_env(path)
        check("the folder round-trips", back.get("DATA_DIR") == str(tmp))
        check("the key round-trips", back.get("GROQ_API_KEY") == "gsk_first")

        # Settings the form never shows, added by hand.
        with io.open(path, "a", encoding="utf-8") as fh:
            fh.write("\n# tuned by hand\nTOP_K=9\nOCR_ENABLED=false\n")
        setup.write_env({"DATA_DIR": str(tmp), "GROQ_API_KEY": "gsk_second"})
        back = setup.read_env(path)
        check("a hand-tuned number survives a save", back.get("TOP_K") == "9")
        check("a hand-tuned flag survives too", back.get("OCR_ENABLED") == "false")
        check("a comment is not eaten",
              "# tuned by hand" in path.read_text(encoding="utf-8"))
        check("the key it does own is updated", back.get("GROQ_API_KEY") == "gsk_second")

        print("\n" + "-" * 64)
        print("a key is never shown in full, and never wiped by silence")
        print("-" * 64)
        masked = setup.mask("gsk_abcdefghijklmnop")
        check("a long key is masked", "abcdefghij" not in masked, masked)
        check("...but stays recognisable", masked.startswith("gsk_"), masked)
        check("a short one gives nothing away",
              set(setup.mask("abc")) <= {"\u2022"}, setup.mask("abc"))
        check("an empty key masks to nothing", setup.mask("") == "")

        # The endpoint carries a blank field through as "keep what is stored"; this
        # is the file-level half of that promise.
        stored = setup.read_env(path).get("GROQ_API_KEY", "")
        setup.write_env({"DATA_DIR": str(tmp), "GROQ_API_KEY": stored})
        check("re-saving with the stored key keeps it",
              setup.read_env(path).get("GROQ_API_KEY") == "gsk_second")

        print("\n" + "-" * 64)
        print("the state the screen draws itself from")
        print("-" * 64)
        st = setup.state()
        for key in ("configured", "reasons", "settings_file", "data_dir",
                    "storage_dir", "groq_key", "gemini_key", "model"):
            check(f"state carries {key}", key in st)
        check("reasons is a list", isinstance(st.get("reasons"), list))
        check("no raw key is in the state",
              "gsk_second" not in str(st), str(st.get("groq_key")))

        print("\n" + "-" * 64)
        print("Ollama is gone")
        print("-" * 64)
        from app import config
        check("no Ollama URL in config", not hasattr(config, "OLLAMA_URL"))
        from app.core import llm
        src = io.open(ROOT / "app" / "core" / "llm.py", encoding="utf-8").read()
        check("no Ollama code path in llm.py", "ollama" not in src.lower())
        emb = io.open(ROOT / "app" / "core" / "embed.py", encoding="utf-8").read()
        check("no Ollama embedder", "OllamaEmbedder" not in emb)
        check("Gemini is a first-class provider",
              hasattr(config, "GEMINI_API_KEY") and hasattr(config, "GEMINI_BASE_URL"))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("RAG_CONFIG_DIR", None)

    print("\n" + "-" * 64)
    if _failures:
        print(f"{len(_failures)} FAILED: " + ", ".join(_failures))
        return 1
    print("all setup tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
