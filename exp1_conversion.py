import json
import os

from deeppavlov import build_model, train_model

from constants import DOMAIN_OF_INTEREST, DATA_DIR, BOT_DATA_DIR

from utils import get_dialogue_files_list, configure_db, initialize_slotfill_config_w_paths, extract_metainfo, \
    train_test_val_split, train_test_val_write, base_conversion, add_api_calls, clear_data_dirs, mockify_dialogues, \
    setup_gobot_config

clear_data_dirs()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BOT_DATA_DIR, exist_ok=True)

dialogues_files = get_dialogue_files_list()
dialogue_files_readt = [(f_split, json.load(open(fn))) for f_split, fn in dialogues_files]
restaurant_dialogues = [(f_split, dialogue)
                        for f_split, dialogues_file in dialogue_files_readt
                        for dialogue in dialogues_file
                        if dialogue['services'] == [DOMAIN_OF_INTEREST]]


dialogues_data, slot_entries = base_conversion(restaurant_dialogues)
mock_dialogues_data, mock_slots_dialogues_data = mockify_dialogues(dialogues_data, slot_entries)
mock_dialogues_w_api, mock_dialogues_w_api_slot_entries = add_api_calls(mock_dialogues_data, mock_slots_dialogues_data)

good_slotfill_dict, templates = extract_metainfo(mock_dialogues_w_api, mock_dialogues_w_api_slot_entries)
initialize_slotfill_config_w_paths(evaluate=False)
unique_slot_names = list(good_slotfill_dict.keys())

database = configure_db(mock_dialogues_w_api)
db_pkeys = database.primary_keys

mock_dialogues_w_api_train, mock_dialogues_w_api_test, mock_dialogues_w_api_dev = train_test_val_split(mock_dialogues_w_api)
train_test_val_write(mock_dialogues_w_api_train, mock_dialogues_w_api_test, mock_dialogues_w_api_dev)

gobot_config = setup_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)

