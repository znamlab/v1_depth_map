PROJECT = "hey2_3d-vision_foodres_20220101"
EYE_TRACKING_SESSIONS = [
    ("PZAH6.4b", "S20220419"),
    ("PZAG3.4f", "S20220524"),
]
MOUSE_LIST = [
    "PZAH6.4b",
    "PZAG3.4f",
    "PZAH8.2i",
    "PZAH8.2h",
    "PZAH8.2f",
    "PZAH10.2d",
    "PZAH10.2f",
]

if False:
    """
    Original old way of handpicking sessions. Kept to check that they are all included in the new way.
    """
    HANDPICKED_SESSIONS = {
        "PZAH6.4b": [
            "220419",
            "220421",
            "220426",
            "220428",
            "220429",
            "220503",
            "220505",
            "220506",
            "220510",
            "220511",
            "220512",
            "220516",
            "220517",
            "220519",
            "220523",
            "220524",
            "220526",
        ],
        "PZAG3.4f": [
            "220419",
            "220421",
            "220422",
            "220426",
            "220429",
            "220503",
            "220504",
            "220505",
            "220508",
            "220509",
            "220510",
            "220511",
            "220512",
            # "220517",  weird
            "220519",
            "220520",
            "220523",
            "220524",
            "220526",
            "220527",
        ],
    }
