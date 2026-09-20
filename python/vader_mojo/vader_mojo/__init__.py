"""vader-mojo: a drop-in faster replacement for the `vaderSentiment` package.

Same `polarity_scores` contract, same scores — powered by a Mojo kernel where
the platform supports it (macOS arm64, Linux x86_64), with a vendored
pure-Python fallback everywhere else (including Windows).

    import vader_mojo

    vader_mojo.polarity_scores("VADER is smart, handsome, and funny!")
    # {'neg': 0.0, 'neu': 0.248, 'pos': 0.752, 'compound': 0.8439}

    from vader_mojo import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()   # drop-in for vaderSentiment's
    analyzer.polarity_scores("Today SUX!")

Set VADER_MOJO_DISABLE_NATIVE=1 to force the pure-Python fallback.

The sentiment rule set and lexicon data are VADER (Hutto & Gilbert, 2014,
ICWSM-14), created by C.J. Hutto and distributed under the MIT license — see
vader_mojo/data/LICENSE.vaderSentiment.txt.
"""

from vader_mojo._native import backend_info, native_available
from vader_mojo.core import SentimentIntensityAnalyzer, polarity_scores

__version__ = "0.1.0"  # x-release-please-version
__all__ = [
    "SentimentIntensityAnalyzer",
    "polarity_scores",
    "backend_info",
    "native_available",
    "__version__",
]
