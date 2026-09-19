"""Differential tests: langdetect_mojo must match PyPI langdetect exactly.

The oracle is the published PyPI package (langdetect==1.0.9), made
deterministic with ``DetectorFactory.seed = 0`` (its default seed is
non-deterministic). langdetect_mojo is deterministic under the same seed by
construction: both sides reproduce the reference's seeded sampling procedure.

The suite runs twice via ``scripts/test_all_langdetect.sh``: once against the
native Mojo kernel and once with LANGDETECT_MOJO_DISABLE_NATIVE=1 (forced
pure-Python fallback). On both backends the outputs must be bit-identical to
the oracle: detect() strings, detect_langs() string forms, and every raw
probability. In practice agreement is bit-exact (0 ulp) on macOS and Linux;
the assertions here are exact equality, not tolerance-based.
"""

from __future__ import annotations

import os

import pytest
from langdetect.detector_factory import DetectorFactory

import langdetect
import langdetect_mojo
from langdetect_mojo import LangDetectError

# One representative text per shipped profile (55 languages).
SAMPLES: dict[str, str] = {
    "af": "Die kinders speel lekker in die tuin saam met hulle vriende.",
    "ar": "هذا نص عربي بسيط لاختبار خاصية التعرف على اللغة في البرنامج.",
    "bg": "Това е примерен текст на български език за проверка на разпознаването.",
    "bn": "এটি বাংলা ভাষায় লেখা একটি সাধারণ পরীক্ষার বাক্য।",
    "ca": "Aquesta és una frase de prova escrita en català per al detector.",
    "cs": "Toto je ukázková česká věta pro testování rozpoznávání jazyka.",
    "cy": "Mae hon yn frawddeg syml yng Nghymraeg i brofi'r canfod iaith.",
    "da": "Dette er en dansk prøvesætning til at teste sproggenkendelsen.",
    "de": "Dies ist ein deutscher Beispielsatz für die Spracherkennung.",
    "el": "Αυτή είναι μια ελληνική πρόταση για τη δοκιμή αναγνώρισης γλώσσας.",
    "en": "This is a sample English sentence for language detection.",
    "es": "Esta es una frase de ejemplo en español para la detección de idiomas.",
    "et": "See on eesti keeles kirjutatud näidislause keele tuvastamiseks.",
    "fa": "این یک جمله نمونه به زبان فارسی برای آزمایش تشخیص زبان است.",
    "fi": "Tämä on suomenkielinen esimerkkilause kielen tunnistusta varten.",
    "fr": "Bonjour le monde, ceci est un texte francais pour le test.",
    "gu": "આ ભાષા ઓળખવા માટે ગુજરાતીમાં લખાયેલો એક નમૂનારૂપ વાક્ય છે.",
    "he": "זהו משפט לדוגמה בעברית לבדיקת זיהוי השפה של הטקסט.",
    "hi": "यह भाषा पहचान की जाँच के लिए हिंदी में लिखा गया एक नमूना वाक्य है।",
    "hr": "Ovo je primjer hrvatske rečenice za testiranje prepoznavanja jezika.",
    "hu": "Ez egy magyar próbamondat a nyelvfelismerő rendszer tesztelésére.",
    "id": "Ini adalah kalimat contoh dalam bahasa Indonesia untuk deteksi bahasa.",
    "it": "Questa è una frase di esempio in italiano per il rilevamento della lingua.",
    "ja": "これは日本語のテキストです。言語検出のテストをしています。",
    "kn": "ಇದು ಭಾಷೆಯನ್ನು ಗುರುತಿಸುವ ಪರೀಕ್ಷೆಗಾಗಿ ಕನ್ನಡದಲ್ಲಿ ಬರೆದ ಮಾದರಿ ವಾಕ್ಯ.",
    "ko": "이것은 언어 감지를 테스트하기 위한 한국어 샘플 문장입니다.",
    "lt": "Tai yra lietuviškas pavyzdinis sakinys kalbos atpažinimui patikrinti.",
    "lv": "Šis ir latviešu valodas paraugteksts valodas noteikšanas pārbaudei.",
    "mk": "Ова е примерна реченица на македонски јазик за тестирање на детекцијата.",
    "ml": "ഇത് ഭാഷ തിരിച്ചറിയുന്നതിനുള്ള മലയാളത്തിൽ എഴുതിയ ഒരു മാതൃകാ വാക്യമാണ്.",
    "mr": "हा भाषा ओळखण्याच्या चाचणीसाठी मराठीत लिहिलेला एक नमुना वाक्य आहे.",
    "ne": "यो भाषा पहिचान परीक्षणका लागि नेपालीमा लेखिएको नमूना वाक्य हो।",
    "nl": "Dit is een Nederlandse voorbeeldzin voor het testen van taalherkenning.",
    "no": "Dette er en norsk prøvesetning for å teste språkgjenkjennelsen.",
    "pa": "ਇਹ ਭਾਸ਼ਾ ਪਛਾਣ ਦੀ ਜਾਂਚ ਲਈ ਪੰਜਾਬੀ ਵਿੱਚ ਲਿਖਿਆ ਇੱਕ ਨਮੂਨਾ ਵਾਕ ਹੈ।",
    "pl": "To jest przykładowe polskie zdanie do testowania wykrywania języka.",
    "pt": "Esta é uma frase de exemplo em português para a detecção de idiomas.",
    "ro": "Aceasta este o propoziție românească pentru testarea detectării limbii.",
    "ru": "Это русский текст для проверки определения языка программой.",
    "sk": "Toto je ukážková slovenská veta na testovanie rozpoznávania jazyka.",
    "sl": "To je vzorčni slovenski stavek za preizkus zaznavanja jezika.",
    "so": "Kani waa jumlad tijaabo ah oo af Soomaali ah loogu talagalay baaritaanka.",
    "sq": "Kjo është një fjali shembull në gjuhën shqipe për testimin e gjuhës.",
    "sv": "Detta är en svensk exempelmening för att testa språkigenkänningen.",
    "sw": "Huu ni mfano wa sentensi ya Kiswahili kwa ajili ya kugundua lugha.",
    "ta": "இது மொழி கண்டறிதலைச் சோதிக்க தமிழில் எழுதப்பட்ட மாதிரி வாக்கியம்.",
    "te": "ఇది భాషా గుర్తింపు పరీక్ష కోసం తెలుగులో వ్రాసిన నమూనా వాక్యం.",
    "th": "นี่คือประโยคตัวอย่างภาษาไทยสำหรับทดสอบการตรวจจับภาษา",
    "tl": "Ito ay isang halimbawang pangungusap sa Tagalog para sa pagsubok.",
    "tr": "Bu, dil algılama testi için yazılmış örnek bir Türkçe cümledir.",
    "uk": "Це зразкове українське речення для перевірки розпізнавання мови.",
    "ur": "یہ زبان کی شناخت کی جانچ کے لیے اردو میں لکھا گیا ایک نمونہ جملہ ہے۔",
    "vi": "Đây là một câu mẫu bằng tiếng Việt để kiểm tra nhận dạng ngôn ngữ.",
    "zh-cn": "这是一个用简体中文写的例句，用于测试语言检测功能。",
    "zh-tw": "這是一個用繁體中文寫的例句，用於測試語言檢測功能。",
}

# Adversarial inputs: preprocessing, normalization, error paths.
EDGE_CASES = [
    "",  # empty
    "   ",  # whitespace only
    "12345 !!!",  # no features
    "?!.,;-'\"",  # punctuation only
    "a",  # single letter
    "z",  # single letter, other end
    "😀😀😀😀😀",  # emoji only: no profile features
    "hello",  # one short word
    "the",  # stopword only
    "ok",  # ambiguous short
    "NASA IS A GREAT AGENCY OF THE UNITED STATES",  # capitalword suppression
    "ΑΘΗΝΑ ΕΙΝΑΙ Η ΠΡΩΤΕΥΟΥΣΑ",  # all-caps Greek
    "МОСКВА СТОЛИЦА РОССИИ",  # all-caps Cyrillic
    "http://example.com",  # URL only, no text
    "user@example.com",  # e-mail only
    "see http://example.com/x?a=b&c=d for details on the project today",  # URL masked
    "write to john.doe@example.com or jane_doe@sub.example-site.org now please",  # mails
    "a" * 100 + "@" + "b" * 300 + ".com",  # mail quantifier backtracking
    "http://http://example.com",  # nested scheme
    "Welcome to https://sub.domain.co.uk/path/page.html?x=1#top everyone!!",  # long URL
    "Hello 世界 this mixes latin and CJK characters in one text",  # cleaning_text
    "東京は日本の首都です Tokyo is the capital",  # heavy non-Latin
    "한국어와 English를 섞어 쓴 문장입니다",  # Hangul + Latin mix
    "Tiếng Việt là ngôn ngữ của ngườngườ Việt",  # precomposed Vietnamese
    "Tieng Viet la ngon ngu cua nguoi Viet",  # Vietnamese without diacritics
    "Toi yeu thien nhien va con nguoi Viet Nam",  # more Vietnamese
    "le renard brun rapide saute par-dessus le chien paresseux",  # fr, lowercase ASCII
    "the quick brown fox jumps over the lazy dog again and again",  # en pangram
    "wpłynąć na przeżycie — zrozumieć źródło",  # pl with dash
    "これはペンです。それは本です。あれは机です。",  # ja repeated script chars
    "今天天气很好，我们去公园散步，然后回家吃饭。",  # zh-cn longer
    "今天天氣很好，我們去公園散步，然後回家吃飯。",  # zh-tw longer
    "Suíomh idirlín https://ga.ie agus ríomhphost eile anseo freisin",  # mixed masks
    "¨¨¨ ~~~ ﷺ",  # symbols/specials blocks
    "Hello  world,  this  has    many    spaces    everywhere.",  # space collapse
    "hello\u0301 world",  # stray combining mark
]

LONG_TEXT = (
    "Language detection is an important building block for text processing. "
    "It decides which language a document or a paragraph is written in, and "
    "it does so by comparing character n-gram statistics against profiles. "
) * 60  # > 10000 codepoints: exercises the max_text_length truncation

SEEDS = [0, 1, 42, 123456789, 2**40 + 5]
SEED_TEXTS = [
    SAMPLES["en"],
    SAMPLES["fr"],
    SAMPLES["ja"],
    SAMPLES["vi"],
    "short text",
    "un texte court en francais",
]


@pytest.fixture(autouse=True)
def _seed_oracle():
    """Every oracle call in this suite is made deterministic (seed 0)."""
    DetectorFactory.seed = 0


def _expected_backend() -> str:
    # scripts/test_all_langdetect.sh runs the suite once per backend.
    return "fallback" if os.environ.get("LANGDETECT_MOJO_DISABLE_NATIVE") == "1" else "native"


def _oracle(text: str) -> tuple[str, list, str]:
    """Oracle (detect, detect_langs, detect_langs str) or the raised message."""
    try:
        dl = langdetect.detect_langs(text)
        return langdetect.detect(text), dl, str(dl)
    except langdetect.LangDetectException as exc:
        return None, None, str(exc)


def _ours(text: str) -> tuple[str, list, str]:
    try:
        dl = langdetect_mojo.detect_langs(text)
        return langdetect_mojo.detect(text), dl, str(dl)
    except LangDetectError as exc:
        return None, None, str(exc)


def _assert_same_output(text: str) -> None:
    ref_detect, ref_langs, ref_str = _oracle(text)
    our_detect, our_langs, our_str = _ours(text)
    assert our_str == ref_str, f"output mismatch for {text[:60]!r}:\nref : {ref_str}\nours: {our_str}"
    if ref_langs is not None:
        assert our_detect == ref_detect
        # raw probabilities are bit-identical, not just their string forms
        assert [x.lang for x in our_langs] == [x.lang for x in ref_langs]
        assert [x.prob for x in our_langs] == [x.prob for x in ref_langs]


def test_backend_is_the_expected_one():
    assert langdetect_mojo.backend() == _expected_backend()


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_detect_parity_all_55_languages(lang):
    _assert_same_output(SAMPLES[lang])


@pytest.mark.parametrize("idx", range(len(EDGE_CASES)))
def test_edge_cases(idx):
    _assert_same_output(EDGE_CASES[idx])


def test_long_text_truncation():
    assert len(LONG_TEXT) > 10000
    _assert_same_output(LONG_TEXT)


def test_lone_surrogate_and_wtf8():
    # Python str can hold lone surrogates; both sides must process identically.
    _assert_same_output("hello \ud800 world this is english text")
    _assert_same_output("\udfff\ud800")


def test_multiple_seeds():
    for seed in SEEDS:
        DetectorFactory.seed = seed
        langdetect_mojo.set_seed(seed)
        try:
            for text in SEED_TEXTS:
                _assert_same_output(text)
        finally:
            DetectorFactory.seed = 0
            langdetect_mojo.set_seed(0)


def test_negative_seed_matches_abs_seed():
    # CPython random.Random(-n) seeds exactly like random.Random(n).
    DetectorFactory.seed = -42
    langdetect_mojo.set_seed(-42)
    try:
        _assert_same_output(SAMPLES["de"])
    finally:
        DetectorFactory.seed = 0
        langdetect_mojo.set_seed(0)


def test_repeat_calls_are_deterministic():
    text = SAMPLES["pt"]
    first = str(langdetect_mojo.detect_langs(text))
    for _ in range(3):
        assert str(langdetect_mojo.detect_langs(text)) == first
        assert langdetect_mojo.detect(text) == langdetect_mojo.detect(text)


def test_error_contract():
    # Same message and numeric code as the oracle's LangDetectException.
    for call in (langdetect.detect, langdetect.detect_langs):
        with pytest.raises(langdetect.LangDetectException) as ref_exc:
            call("!!!")
        assert str(ref_exc.value) == "No features in text."
        assert ref_exc.value.get_code() == 5
    for call in (langdetect_mojo.detect, langdetect_mojo.detect_langs):
        with pytest.raises(LangDetectError) as ours_exc:
            call("!!!")
        assert str(ours_exc.value) == "No features in text."
        assert ours_exc.value.code == 5
        assert ours_exc.value.get_code() == 5


def test_input_validation():
    with pytest.raises(TypeError):
        langdetect_mojo.detect(123)
    with pytest.raises(TypeError):
        langdetect_mojo.detect(None)
    with pytest.raises(TypeError):
        langdetect_mojo.set_seed("0")
    with pytest.raises(ValueError):
        langdetect_mojo.set_seed(1 << 80)
    # oracle raises TypeError on non-str as well (re.sub on an int)
    with pytest.raises(TypeError):
        langdetect.detect(123)
