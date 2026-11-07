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

# Telugu IPA phonetic inventory from Epitran and MFA alignments
_raw_telugu_ipa = [
    "a", "aː", "b", "b̤", "d", "d̪", "d̪̤", "d͡ʒ", "e", "eː", "f", "h", "i", "iː", "j", "k", "kʰ",
    "l", "m", "n", "o", "oː", "p", "pʰ", "r", "s", "t", "t̪", "t̪ʰ", "u", "uː", "ŋ", "ɖ", "ɖ̤",
    "ə", "ɡ", "ɡ̤", "ɪ", "ɭ", "ɲ", "ɳ", "ʂ", "ʃ", "ʃʰ", "ʈ", "ʈʰ", "ʊ", "ʋ", "ʌ", "ʒ", "ʒ̤", "͡"
]
_telugu_ipa = ["@" + s for s in _raw_telugu_ipa]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _telugu_characters
    + _telugu_ipa
    + _arpabet
    + _pinyin
    + _silences
)
