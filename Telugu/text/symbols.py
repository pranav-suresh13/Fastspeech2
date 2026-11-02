"""
Defines the set of symbols used in text input to the model for Telugu.
Includes standard punctuation, special tokens, English ASCII, silences,
and the full Telugu Unicode character set (\u0C00 - \u0C7F).
"""

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? \n\"'’“”"
_special = "-_~"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Telugu Unicode range (0x0C00 to 0x0C7F)
_telugu_characters = [chr(cp) for cp in range(0x0C00, 0x0C80) if chr(cp).isprintable()]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _telugu_characters
    + _arpabet
    + _pinyin
    + _silences
)
