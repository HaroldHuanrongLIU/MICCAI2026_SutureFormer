"""Patient-level train/validation/test split definitions."""

from typing import List

TRAIN_PATIENTS: List[str] = [
    "006",
    "007",
    "010",
    "011",
    "012",
    "014",
    "019",
    "020",
    "022",
    "026",
    "028",
    "029",
    "031",
    "036",
    "037",
    "038",
    "039",
    "040",
    "045",
    "050",
    "052",
    "054",
    "055",
    "056",
    "057",
    "061",
    "064",
    "065",
    "067",
    "068",
    "071",
    "072",
    "073",
    "074",
    "702",
]

VAL_PATIENTS: List[str] = [
    "023",
    "024",
    "030",
    "046",
    "053",
    "069",
    "076",
    "101",
]

TEST_PATIENTS: List[str] = [
    "021",
    "058",
    "075",
    "077",
    "078",
    "100",
    "701",
]


def get_split(split: str) -> List[str]:
    """Return list of patient IDs for the given split.

    Args:
        split: One of 'train', 'val', 'test'.

    Returns:
        List of patient ID strings.
    """
    return {"train": TRAIN_PATIENTS, "val": VAL_PATIENTS, "test": TEST_PATIENTS}[split]
