#!/usr/bin/env python3
"""Reproducible benchmark: PyPI langdetect vs langdetect_mojo (native + fallback).

Correctness is gated first: every corpus text must produce bit-identical
output (detect string and full detect_langs probability values) on the native
and fallback backends, or the benchmark aborts. Timings:

- cold first-call: wall time of (interpreter start already excluded) import +
  profile load + first detect() in a fresh process, median of 5 launches.
- warm steady-state: per-call latency over the benchmark corpus, median of 5
  batches, for the oracle, the native kernel and the forced fallback.

The corpus is the 55-language sample set from the differential suite plus a
>10000-codepoint text (exercises truncation) — no network, no datasets.

Run from the repository root, e.g.:

    PYTHONPATH=python/langdetect_mojo /tmp/langdetect-mojo-venv/bin/python benchmarks/bench_langdetect.py
"""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import sys
import time

N_RUNS = 5  # median over this many batches / launches
SEED = 0

# 55-language corpus: one representative text per profile (kept in sync with
# tests/test_langdetect_differential.py; duplicated so the benchmark stands alone).
SAMPLES = [
    "Die kinders speel lekker in die tuin saam met hulle vriende.",
    "هذا نص عربي بسيط لاختبار خاصية التعرف على اللغة في البرنامج.",
    "Това е примерен текст на български език за проверка на разпознаването.",
    "এটি বাংলা ভাষায় লেখা একটি সাধারণ পরীক্ষার বাক্য।",
    "Aquesta és una frase de prova escrita en català per al detector.",
    "Toto je ukázková česká věta pro testování rozpoznávání jazyka.",
    "Mae hon yn frawddeg syml yng Nghymraeg i brofi'r canfod iaith.",
    "Dette er en dansk prøvesætning til at teste sproggenkendelsen.",
    "Dies ist ein deutscher Beispielsatz für die Spracherkennung.",
    "Αυτή είναι μια ελληνική πρόταση για τη δοκιμή αναγνώρισης γλώσσας.",
    "This is a sample English sentence for language detection.",
    "Esta es una frase de ejemplo en español para la detección de idiomas.",
    "See on eesti keeles kirjutatud näidislause keele tuvastamiseks.",
    "این یک جمله نمونه به زبان فارسی برای آزمایش تشخیص زبان است.",
    "Tämä on suomenkielinen esimerkkilause kielen tunnistusta varten.",
    "Bonjour le monde, ceci est un texte francais pour le test.",
    "આ ભાષા ઓળખવા માટે ગુજરાતીમાં લખાયેલો એક નમૂનારૂપ વાક્ય છે.",
    "זהו משפט לדוגמה בעברית לבדיקת זיהוי השפה של הטקסט.",
    "यह भाषा पहचान की जाँच के लिए हिंदी में लिखा गया एक नमूना वाक्य है।",
    "Ovo je primjer hrvatske rečenice za testiranje prepoznavanja jezika.",
    "Ez egy magyar próbamondat a nyelvfelismerő rendszer tesztelésére.",
    "Ini adalah kalimat contoh dalam bahasa Indonesia untuk deteksi bahasa.",
    "Questa è una frase di esempio in italiano per il rilevamento della lingua.",
    "これは日本語のテキストです。言語検出のテストをしています。",
    "ಇದು ಭಾಷೆಯನ್ನು ಗುರುತಿಸುವ ಪರೀಕ್ಷೆಗಾಗಿ ಕನ್ನಡದಲ್ಲಿ ಬರೆದ ಮಾದರಿ ವಾಕ್ಯ.",
    "이것은 언어 감지를 테스트하기 위한 한국어 샘플 문장입니다.",
    "Tai yra lietuviškas pavyzdinis sakinys kalbos atpažinimui patikrinti.",
    "Šis ir latviešu valodas paraugteksts valodas noteikšanas pārbaudei.",
    "Ова е примерна реченица на македонски јазик за тестирање на детекцијата.",
    "ഇത് ഭാഷ തിരിച്ചറിയുന്നതിനുള്ള മലയാളത്തിൽ എഴുതിയ ഒരു മാതൃകാ വാക്യമാണ്.",
    "हा भाषा ओळखण्याच्या चाचणीसाठी मराठीत लिहिलेला एक नमुना वाक्य आहे.",
    "यो भाषा पहिचान परीक्षणका लागि नेपालीमा लेखिएको नमूना वाक्य हो।",
    "Dit is een Nederlandse voorbeeldzin voor het testen van taalherkenning.",
    "Dette er en norsk prøvesetning for å teste språkgjenkjennelsen.",
    "ਇਹ ਭਾਸ਼ਾ ਪਛਾਣ ਦੀ ਜਾਂਚ ਲਈ ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖਿਆ ਇੱਕ ਨਮੂਨਾ ਵਾਕ ਹੈ।",
    "To jest przykładowe polskie zdanie do testowania wykrywania języka.",
    "Esta é uma frase de exemplo em português para a detecção de idiomas.",
    "Aceasta este o propoziție românească pentru testarea detectării limbii.",
    "Это русский текст для проверки определения языка программой.",
    "Toto je ukážková slovenská veta na testovanie rozpoznávania jazyka.",
    "To je vzorčni slovenski stavek za preizkus zaznavanja jezika.",
    "Kani waa jumlad tijaabo ah oo af Soomaali ah loogu talagalay baaritaanka.",
    "Kjo është një fjali shembull në gjuhën shqipe për testimin e gjuhës.",
    "Detta är en svensk exempelmening för att testa språkigenkänningen.",
    "Huu ni mfano wa sentensi ya Kiswahili kwa ajili ya kugundua lugha.",
    "இது மொழி கண்டறிதலைச் சோதிக்க தமிழில் எழுதப்பட்ட மாதிரி வாக்கியம்.",
    "ఇది భాషా గుర్తింపు పరీక్ష కోసం తెలుగులో వ్రాసిన నమూనా వాక్యం.",
    "นี่คือประโยคตัวอย่างภาษาไทยสำหรับทดสอบการตรวจจับภาษา",
    "Ito ay isang halimbawang pangungusap sa Tagalog para sa pagsubok.",
    "Bu, dil algılama testi için yazılmış örnek bir Türkçe cümledir.",
    "Це зразкове українське речення для перевірки розпізнавання мови.",
    "یہ زبان کی شناخت کی جانچ کے لیے اردو میں لکھا گیا ایک نمونہ جملہ ہے۔",
    "Đây là một câu mẫu bằng tiếng Việt để kiểm tra nhận dạng ngôn ngữ.",
    "这是一个用简体中文写的例句，用于测试语言检测功能。",
    "這是一個用繁體中文寫的例句，用於測試語言檢測功能。",
]

LONG_TEXT = (
    "Language detection is an important building block for text processing. "
    "It decides which language a document or a paragraph is written in, and "
    "it does so by comparing character n-gram statistics against profiles. "
) * 60  # > 10000 codepoints

COLD_SNIPPETS = {
    "oracle": (
        "import time; t0=time.perf_counter();"
        "import langdetect;"
        "from langdetect.detector_factory import DetectorFactory;"
        "DetectorFactory.seed=0;"
        "langdetect.detect('This is a sample English sentence for language detection.');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "native": (
        "import time; t0=time.perf_counter();"
        "import langdetect_mojo;"
        "langdetect_mojo.detect('This is a sample English sentence for language detection.');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
    "fallback": (
        "import time; t0=time.perf_counter();"
        "import langdetect_mojo;"
        "langdetect_mojo.detect('This is a sample English sentence for language detection.');"
        "print(f'{time.perf_counter()-t0:.6f}')"
    ),
}


def cold_start(which: str) -> float:
    """Median seconds for import + first detect in a fresh process."""
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    if which == "fallback":
        env["LANGDETECT_MOJO_DISABLE_NATIVE"] = "1"
    samples = []
    for _ in range(N_RUNS):
        out = subprocess.run(
            [sys.executable, "-c", COLD_SNIPPETS[which]],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        samples.append(float(out.stdout.strip()))
    return statistics.median(samples)


def machine_info() -> str:
    lines = [
        f"- date: {time.strftime('%Y-%m-%d')}",
        f"- machine: {platform.platform()} ({platform.machine()})",
    ]
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        if chip:
            lines.append(f"- cpu: {chip}")
    except OSError:
        pass
    lines.append(f"- python: {platform.python_version()}")
    try:
        mojo = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True
        ).stdout.strip()
        lines.append(f"- mojo: {mojo}")
    except OSError:
        pass
    return "\n".join(lines)


def main() -> None:
    from langdetect.detector_factory import DetectorFactory

    import langdetect
    import langdetect_mojo

    DetectorFactory.seed = SEED
    langdetect_mojo.set_seed(SEED)

    info = langdetect_mojo.backend_info()
    print("== environment ==")
    print(machine_info())
    print(
        f"- langdetect_mojo backend: {'native' if info['native_available'] else 'FALLBACK'} "
        f"({info.get('native_source') or info.get('error')})"
    )
    print(f"- corpus: {len(SAMPLES)} texts (55 languages) + 1 long text ({len(LONG_TEXT)} chars)")
    print(f"- runs: median of {N_RUNS}")
    if not info["native_available"]:
        sys.exit("native kernel unavailable; refusing to benchmark the fallback as 'native'")

    print("\n== correctness gate (bit-exact vs seeded oracle) ==")
    corpus = SAMPLES + [LONG_TEXT]
    for disable, label in ((False, "native"), (True, "fallback")):
        os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "1" if disable else "0"
        worst = 0
        for text in corpus:
            ref = (langdetect.detect(text), str(langdetect.detect_langs(text)))
            ours = (langdetect_mojo.detect(text), str(langdetect_mojo.detect_langs(text)))
            if ours != ref:
                sys.exit(f"correctness gate failed ({label}) for {text[:50]!r}: {ours} != {ref}")
        print(f"  {label}: all {len(corpus)} texts bit-identical")
    os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "0"

    def time_calls(fn, texts, inner_repeats=1) -> float:
        """Median seconds for one full pass over `texts`."""
        samples = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            for _ in range(inner_repeats):
                for text in texts:
                    fn(text)
            samples.append((time.perf_counter() - t0) / inner_repeats)
        return statistics.median(samples)

    print("\n== warm steady-state: detect() latency over the 55-text corpus ==")
    t_oracle = time_calls(langdetect.detect, SAMPLES)
    t_native = time_calls(langdetect_mojo.detect, SAMPLES)
    os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "1"
    t_fallback = time_calls(langdetect_mojo.detect, SAMPLES)
    os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "0"
    per_oracle, per_native, per_fallback = (1e6 * t / len(SAMPLES) for t in (t_oracle, t_native, t_fallback))
    print(f"  oracle (PyPI langdetect): {per_oracle:8.1f} µs/call")
    print(f"  langdetect_mojo native:   {per_native:8.1f} µs/call  ({per_oracle/per_native:.1f}x)")
    print(f"  langdetect_mojo fallback: {per_fallback:8.1f} µs/call  ({per_oracle/per_fallback:.2f}x)")

    print("\n== warm steady-state: detect() on the long (>10000 chars) text ==")
    lo = time_calls(langdetect.detect, [LONG_TEXT])
    ln = time_calls(langdetect_mojo.detect, [LONG_TEXT])
    os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "1"
    lf = time_calls(langdetect_mojo.detect, [LONG_TEXT])
    os.environ["LANGDETECT_MOJO_DISABLE_NATIVE"] = "0"
    print(f"  oracle:  {1e3*lo:8.3f} ms/call")
    print(f"  native:  {1e3*ln:8.3f} ms/call  ({lo/ln:.1f}x)")
    print(f"  fallback:{1e3*lf:8.3f} ms/call  ({lo/lf:.2f}x)")

    print("\n== cold first-call (fresh process: import + profiles + first detect) ==")
    c_oracle = cold_start("oracle")
    c_native = cold_start("native")
    c_fallback = cold_start("fallback")
    print(f"  oracle:  {1e3*c_oracle:8.1f} ms")
    print(f"  native:  {1e3*c_native:8.1f} ms  ({c_oracle/c_native:.2f}x)")
    print(f"  fallback:{1e3*c_fallback:8.1f} ms  ({c_oracle/c_fallback:.2f}x)")

    print("\n== README paste block ==")
    print("Warm steady-state `detect()` (median of 5):")
    print()
    print("| workload | PyPI langdetect | langdetect-mojo (native) | langdetect-mojo (fallback) | native speedup |")
    print("|---|---:|---:|---:|---:|")
    print(f"| 55-language corpus, per call | {per_oracle:.1f} µs | {per_native:.1f} µs | {per_fallback:.1f} µs | {per_oracle/per_native:.1f}x |")
    print(f"| long text ({len(LONG_TEXT)} chars, truncated to 10000), per call | {1e3*lo:.3f} ms | {1e3*ln:.3f} ms | {1e3*lf:.3f} ms | {lo/ln:.1f}x |")
    print()
    print("Cold first-call (fresh process, median of 5):")
    print()
    print("| package | import + first detect |")
    print("|---|---:|")
    print(f"| PyPI langdetect (seeded) | {1e3*c_oracle:.1f} ms |")
    print(f"| langdetect-mojo (native) | {1e3*c_native:.1f} ms |")
    print(f"| langdetect-mojo (fallback) | {1e3*c_fallback:.1f} ms |")


if __name__ == "__main__":
    main()
