"""NAICS code → human-readable business category mapping.

2-digit sector codes map to ~20 categories. Sub-category refinement for
key sectors (e.g., 722 food service → restaurant types vs bars).
"""

from __future__ import annotations

# 2-digit NAICS sector → category
NAICS_CATEGORIES: dict[str, str] = {
    "11": "agriculture",
    "21": "mining & extraction",
    "22": "utilities",
    "23": "construction",
    "31": "manufacturing",
    "32": "manufacturing",
    "33": "manufacturing",
    "42": "wholesale trade",
    "44": "retail",
    "45": "retail",
    "48": "transportation",
    "49": "warehousing & logistics",
    "51": "information & media",
    "52": "finance & insurance",
    "53": "real estate",
    "54": "professional services",
    "55": "management of companies",
    "56": "administrative services",
    "61": "education",
    "62": "healthcare",
    "71": "arts & entertainment",
    "72": "food & accommodation",
    "81": "personal services",
    "92": "government",
}

# 3-4 digit sub-categories for key sectors
NAICS_SUBCATEGORIES: dict[str, str] = {
    # food service (72)
    "7223": "food service contractor",
    "7224": "drinking places",
    "7225": "restaurant",
    "72251": "restaurant",
    "722511": "restaurant",      # full-service
    "722513": "restaurant",      # limited-service (fast food/fast casual)
    "722514": "cafe & snack bar",
    "722515": "cafe & snack bar",
    "7221": "food & accommodation",
    "7211": "hotel & lodging",
    "72111": "hotel & lodging",
    "721110": "hotel & lodging",
    "72119": "hotel & lodging",
    "721191": "bed & breakfast",
    # retail sub-categories (44-45)
    "4451": "grocery",
    "44511": "grocery",
    "4452": "specialty food store",
    "4453": "liquor store",
    "4461": "health & personal care",
    "4471": "gas station",
    "4481": "clothing store",
    "4511": "sporting goods",
    "4521": "department store",
    "4531": "florist",
    "4532": "office supply",
    "4533": "used merchandise",
    "4539": "pet store",
    # healthcare (62)
    "6211": "physician",
    "6212": "dentist",
    "6213": "healthcare practitioner",
    "6214": "outpatient care",
    "6215": "medical lab",
    "6216": "home health care",
    "6221": "hospital",
    "6231": "nursing care",
    "6241": "social services",
    "6244": "child care",
    # professional services (54)
    "5411": "legal services",
    "5412": "accounting",
    "5413": "architecture & engineering",
    "5414": "graphic design",
    "5415": "technology services",
    "5416": "management consulting",
    "5417": "scientific research",
    "5418": "advertising & marketing",
    # personal services (81)
    "8111": "auto repair",
    "8112": "electronics repair",
    "8113": "commercial repair",
    "8121": "personal care (salon/spa)",
    "8122": "funeral services",
    "8123": "laundry & dry cleaning",
    "8129": "pet care",
    # arts & entertainment (71)
    "7111": "performing arts",
    "7112": "spectator sports",
    "7121": "museum",
    "7131": "amusement park",
    "7139": "recreation & fitness",
    # construction (23)
    "2361": "residential construction",
    "2362": "commercial construction",
    "2381": "building foundation",
    "2382": "building equipment",
    "2383": "building finishing",
    "2389": "specialty trade",
}

# keyword fallback for descriptions with no usable NAICS code
_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["restaurant", "dining", "eatery", "bistro", "grill", "taqueria", "sushi"], "restaurant"),
    (["coffee", "cafe", "espresso", "tea house"], "cafe & snack bar"),
    (["bar", "pub", "tavern", "brewery", "taproom", "lounge", "nightclub"], "drinking places"),
    (["pizza", "pizzeria"], "restaurant"),
    (["bakery", "donut", "bagel"], "cafe & snack bar"),
    (["salon", "barber", "spa", "beauty", "nail", "hair"], "personal care (salon/spa)"),
    (["gym", "fitness", "yoga", "pilates", "crossfit"], "recreation & fitness"),
    (["dentist", "dental"], "dentist"),
    (["doctor", "physician", "medical", "clinic", "chiropractic"], "physician"),
    (["lawyer", "attorney", "law office", "legal"], "legal services"),
    (["accounting", "accountant", "cpa", "bookkeeping", "tax prep"], "accounting"),
    (["real estate", "property management", "realty"], "real estate"),
    (["insurance"], "finance & insurance"),
    (["hotel", "motel", "inn", "resort", "hostel"], "hotel & lodging"),
    (["grocery", "market", "supermarket"], "grocery"),
    (["auto repair", "mechanic", "body shop", "oil change"], "auto repair"),
    (["daycare", "child care", "preschool"], "child care"),
    (["consulting"], "management consulting"),
    (["construction", "contractor", "plumbing", "electric", "roofing", "hvac"], "construction"),
    (["retail", "shop", "store", "boutique"], "retail"),
    (["tech", "software", "web develop", "it service", "app develop"], "technology services"),
]


def map_naics(code: str | None, description: str = "") -> str:
    """Map a NAICS code (+ optional description) to a human-readable category.

    Tries 4-digit sub-category first, then 3-digit, then 2-digit sector,
    then keyword matching on description. Returns "other" as last resort.
    """
    if code:
        code = str(code).strip().split(".")[0]  # handle floats / decimals
        # try progressively shorter prefixes: full → 5 → 4 → 3 → 2
        for length in (len(code), 5, 4, 3):
            prefix = code[:length]
            if prefix in NAICS_SUBCATEGORIES:
                return NAICS_SUBCATEGORIES[prefix]
        # 2-digit sector
        sector = code[:2]
        if sector in NAICS_CATEGORIES:
            return NAICS_CATEGORIES[sector]

    # keyword fallback on description
    if description:
        desc_lower = description.lower()
        for keywords, category in _KEYWORD_MAP:
            if any(kw in desc_lower for kw in keywords):
                return category

    return "other"
