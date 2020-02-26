import json
import os

from deeppavlov import train_model

from constants import DOMAIN_OF_INTEREST, DATA_DIR, BOT_DATA_DIR
from utils import get_dstc8_dialogue_files_list, configure_db, initialize_slotfill_model_config_w_paths, extract_slotfill_and_templates, \
    train_test_val_split, train_test_val_write, base_dstc8_2_dstc2, add_db_api_calls, clear_data_dirs, configure_gobot_config

clear_data_dirs()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BOT_DATA_DIR, exist_ok=True)

dialogues_files = get_dstc8_dialogue_files_list()
dialogue_files_readt = [(f_split, json.load(open(fn))) for f_split, fn in dialogues_files]
restaurant_dialogues = [(f_split, dialogue)
                        for f_split, dialogues_file in dialogue_files_readt
                        for dialogue in dialogues_file
                        if dialogue['services'] == [DOMAIN_OF_INTEREST]]


dialogues_data, slot_entries = base_dstc8_2_dstc2(restaurant_dialogues)
dialogues_w_api, dialogues_w_api_slot_entries = add_db_api_calls(dialogues_data, slot_entries)

good_slotfill_dict, templates = extract_slotfill_and_templates(dialogues_w_api, dialogues_w_api_slot_entries)
initialize_slotfill_model_config_w_paths(evaluate=False)
unique_slot_names = list(good_slotfill_dict.keys())

database = configure_db(dialogues_w_api)
db_pkeys = database.primary_keys

dialogues_w_api_train, dialogues_w_api_test, dialogues_w_api_dev = train_test_val_split(dialogues_w_api)
train_test_val_write(dialogues_w_api_train, dialogues_w_api_test, dialogues_w_api_dev)


# WARNING! will train for long time because of the huge number of possible slot values

gobot_config = configure_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)
