from dataclasses import dataclass
from typing import Dict

@dataclass
class BanditEnvConfig:
    split: str = "train"
    action_space_start: int = 1
    lo_arm_score: float = 0.1
    hi_arm_loscore: float = 0.0
    hi_arm_hiscore: float = 1.0
    hi_arm_hiscore_prob: float = 0.25
    render_mode: str = "text"
    action_lookup: Dict[int, str] = None # defined in env.py


ARM_NAMES = {
    "train": [
        ("Teacher", "Trader"),
        ("Nurse", "StartupFounder"),
        ("Librarian", "Investor"),
        ("Accountant", "StockBroker"),
        ("Engineer", "RealEstateAgent"),
        ("Pharmacist", "Musician"),
        ("Clerk", "Freelancer"),
        ("Technician", "Youtuber"),
        ("Planner", "Artist"),
        ("Postman", "Pilot"),
        ("Banker", "VentureCapitalist"),
        ("Gardener", "Cryptominer"),
    ],
    "test": [
        ("Receptionist", "Explorer"),
        ("Archivist", "Filmmaker"),
        ("Bookkeeper", "Consultant"),
        ("SocialWorker", "Photographer"),
        ("Mechanic", "GameDesigner"),
        ("Secretary", "Entrepreneur"),
        ("Cashier", "DataScientist"),
        ("CivilServant", "StartupAccelerator"),
    ],
}