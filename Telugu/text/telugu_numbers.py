"""
Telugu number and time expander for text normalization.
"""

import re

TELUGU_DIGITS = {
    "0": "సున్నా",
    "1": "ఒకటి",
    "2": "రెండు",
    "3": "మూడు",
    "4": "నాలుగు",
    "5": "ఐదు",
    "6": "ఆరు",
    "7": "ఏడు",
    "8": "ఎనిమిది",
    "9": "తొమ్మిది",
}

TELUGU_NUMBERS = {
    1: "ఒకటి", 2: "రెండు", 3: "మూడు", 4: "నాలుగు", 5: "ఐదు",
    6: "ఆరు", 7: "ఏడు", 8: "ఎనిమిది", 9: "తొమ్మిది", 10: "పది",
    11: "పదకొండు", 12: "పన్నెండు", 13: "పదమూడు", 14: "పద్నాలుగు", 15: "పదిహేను",
    16: "పదహారు", 17: "పదిహేడు", 18: "పద్దెనిమిది", 19: "పంతొమ్మిది", 20: "ఇరవై",
    21: "ఇరవై ఒకటి", 22: "ఇరవై రెండు", 23: "ఇరవై మూడు", 24: "ఇరవై నాలుగు", 25: "ఇరవై ఐదు",
    26: "ఇరవై ఆరు", 27: "ఇరవై ఏడు", 28: "ఇరవై ఎనిమిది", 29: "ఇరవై తొమ్మిది", 30: "ముప్పై",
    35: "ముప్పై ఐదు", 40: "నలభై", 45: "నలభై ఐదు", 50: "యాభై", 60: "అరవై",
    70: "డెబ్బై", 80: "ఎనభై", 90: "తొంభై", 100: "వంద", 1000: "వెయ్యి"
}


def num_to_telugu(num: int) -> str:
    if num in TELUGU_NUMBERS:
        return TELUGU_NUMBERS[num]
    if num < 100:
        tens = (num // 10) * 10
        rem = num % 10
        tens_str = TELUGU_NUMBERS.get(tens, str(tens))
        rem_str = TELUGU_NUMBERS.get(rem, str(rem))
        return f"{tens_str} {rem_str}"
    if num < 1000:
        hundreds = num // 100
        rem = num % 100
        h_str = f"{TELUGU_NUMBERS.get(hundreds, '')} వందల" if hundreds > 1 else "వంద"
        return f"{h_str} {num_to_telugu(rem)}" if rem > 0 else h_str
    if num < 100000:
        thousands = num // 1000
        rem = num % 1000
        t_str = f"{num_to_telugu(thousands)} వేల" if thousands > 1 else "వెయ్యి"
        return f"{t_str} {num_to_telugu(rem)}" if rem > 0 else t_str
    return " ".join([TELUGU_DIGITS.get(d, d) for d in str(num)])


def normalize_telugu_time(match):
    h = int(match.group(1))
    m = int(match.group(2))
    h_str = num_to_telugu(h)
    m_str = num_to_telugu(m)
    return f"{h_str} గంటల {m_str} నిమిషాలు"


def normalize_telugu_ordinal(match):
    num = int(match.group(1))
    return f"{num_to_telugu(num)}వ"


def normalize_telugu_numbers(text: str) -> str:
    # 1. Normalize time like 10:45
    text = re.sub(r'(\d{1,2}):(\d{2})', normalize_telugu_time, text)
    # 2. Normalize ordinals like 12వ
    text = re.sub(r'(\d+)\s*వ', normalize_telugu_ordinal, text)
    # 3. Normalize standalone numbers (e.g. 2026)
    text = re.sub(r'\b\d+\b', lambda m: num_to_telugu(int(m.group(0))), text)
    return text
