"""The real engines each of the game's engines is drawn after, with a Wikipedia page for each.
The pictures are our own pixel art; these say only where their outline and colours come from."""

WIKI = "https://en.wikipedia.org/wiki/"

ORIGINS: dict[str, tuple[tuple[str, str], ...]] = {
    "penydarren": (("Penydarren locomotive, 1804", "Penydarren_(locomotive)"),),
    "catch_me": (("Catch Me Who Can, 1808", "Catch_Me_Who_Can"),),
    "salamanca": (("Salamanca, 1812", "Salamanca_(locomotive)"),),
    "puffing_billy": (("Puffing Billy, 1813", "Puffing_Billy_(locomotive)"),),
    "locomotion": (("Locomotion No. 1, 1825", "Locomotion_No._1"),),
    "rocket": (("Stephenson's Rocket, 1829", "Stephenson%27s_Rocket"),),
    "lion": (("LMR 57 Lion, 1838", "LMR_57_Lion"),),
    "planet": (("Planet, 1830", "Planet_(locomotive)"),),
    "tank": (("Tank locomotive (type)", "Tank_locomotive"),),
    "goods": (("0-6-0 goods engine (type)", "0-6-0"),),
    "american": (
        ("4-4-0 American (type)", "4-4-0"),
        ("The General, 1855", "The_General_(locomotive)"),
    ),
    "single": (("GNR Stirling 4-2-2, 1870", "GNR_Stirling_4-2-2"),),
    "light_tank": (("0-6-2 tank engine (type)", "0-6-2"),),
    "mogul": (("2-6-0 Mogul (type)", "2-6-0"),),
    "atlantic": (("4-4-2 Atlantic (type)", "4-4-2_(locomotive)"),),
    "mallet": (("Mallet locomotive (type)", "Mallet_locomotive"),),
    "pacific": (("4-6-2 Pacific (type)", "4-6-2"),),
    "boxcab": (("Boxcab electric (type)", "Boxcab"),),
    "railbus": (("Railbus (type)", "Railbus"), ("Uerdingen railbus", "Uerdingen_railbus")),
    "streamliner": (("Streamliner (type)", "Streamliner"), ("LNER Class A4", "LNER_Class_A4")),
    "big_boy": (("Union Pacific Big Boy, 1941", "Union_Pacific_Big_Boy"),),
    "mixed_diesel": (("EMD F7", "EMD_F7"),),
    "heavy_diesel": (("EMD SD40-2", "EMD_SD40-2"),),
    "bullet": (("0 Series Shinkansen, 1964", "0_Series_Shinkansen"),),
    "express": (("DB Class 103", "DB_Class_103"),),
    "sprinter": (("British Rail Class 150", "British_Rail_Class_150"),),
    "high_speed": (("ICE 1", "ICE_1"),),
    "twin_diesel": (("GE Dash 9-44CW", "GE_Dash_9-44CW"),),
    "modern_electric": (("DB Class 101", "DB_Class_101"),),
    "super_diesel": (),
    "maglev": (("Transrapid", "Transrapid"), ("SCMaglev L0 Series", "L0_Series")),
}


def origins(engine: str) -> tuple[tuple[str, str], ...]:
    """The real engines one is drawn after, as names and Wikipedia links; none for our own."""
    return tuple((name, WIKI + page) for name, page in ORIGINS[engine])
