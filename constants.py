import os

"""## constants"""

# region constants

BOT_DATA_DIR = "my_bot"
DATA_DIR = "my_data"
TEMPLATES_FN = "simple-dstc2-templates.txt"
TEMPLATES_FPATH = os.path.join(DATA_DIR, TEMPLATES_FN)
DB_FN = "db.sqlite"
DB_FPATH = os.path.join(BOT_DATA_DIR, DB_FN)
RES_TRN_FN = "simple-dstc2-trn.json"
RES_TRN_FPATH = os.path.join(DATA_DIR, RES_TRN_FN)
RES_TST_FN = "simple-dstc2-tst.json"
RES_TST_FPATH = os.path.join(DATA_DIR, RES_TST_FN)
RES_VAL_FN = "simple-dstc2-val.json"
RES_VAL_FPATH = os.path.join(DATA_DIR, RES_VAL_FN)
SLOTFILL_DATA_FN = "slotfill.json"
SLOTFILL_DATA_FPATH = os.path.join(BOT_DATA_DIR, SLOTFILL_DATA_FN)
SLOTFILL_CONFIG_FN = "slotfill_config.json"
SLOTFILL_CONFIG_FPATH = os.path.join(BOT_DATA_DIR, SLOTFILL_CONFIG_FN)
SPEAKER_2_IDX = {"USER": 1, "SYSTEM": 2}
IDX_2_SPEAKER = {v: k for k, v in SPEAKER_2_IDX.items()}

DSTC8_REPO_PATH = "dstc8-schema-guided-dialogue"
DOMAIN_OF_INTEREST = "Travel_1"

"""
complete list of dstc8 domains:

"Alarm_1",
"Banks_1", 
"Banks_2",
"Buses_1", 
"Buses_2",
"Calendar_1",
"Events_1", 
"Events_2",
"Flights_1", 
"Flights_2", 
"Flights_3",
"Homes_1",
"Hotels_1", 
"Hotels_2", 
"Hotels_3", 
"Hotels_4",
"Media_1", 
"Media_2",
"Movies_1", 
"Movies_2",
"Music_1", 
"Music_2",
"RentalCars_1", 
"RentalCars_2",
"Restaurants_1", 
"Restaurants_2",
"RideSharing_1", 
"RideSharing_2",
"Services_1", 
"Services_2", 
"Services_3", 
"Services_4",
"Travel_1",
"Weather_1"
"""
# endregion constants
TRN_F_AFFIX = "train"
TST_F_AFFIX = "test"
DEV_F_AFFIX = "dev"