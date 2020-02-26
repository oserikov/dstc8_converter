import json
import os

from deeppavlov import build_model, train_model

from constants import DOMAIN_OF_INTEREST, BOT_DATA_DIR, DATA_DIR

from utils import get_dstc8_dialogue_files_list, configure_db, initialize_slotfill_model_config_w_paths, extract_slotfill_and_templates, \
    train_test_val_split, train_test_val_write, base_dstc8_2_dstc2, add_db_api_calls, clear_data_dirs, mockify_slots_in_dialogues, \
    configure_gobot_config

clear_data_dirs()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BOT_DATA_DIR, exist_ok=True)

dialogues_files = get_dstc8_dialogue_files_list(skip_train=False, skip_test=False, skip_val=False)
dialogue_files_readt = [(f_split, json.load(open(fn))) for f_split, fn in dialogues_files]
restaurant_dialogues = [(f_split, dialogue)
                        for f_split, dialogues_file in dialogue_files_readt
                        for dialogue in dialogues_file
                        if dialogue['services'] == [DOMAIN_OF_INTEREST]]

dialogues_data, slot_entries = base_dstc8_2_dstc2(restaurant_dialogues)
mock_dialogues_data, mock_slots_dialogues_data = mockify_slots_in_dialogues(dialogues_data, slot_entries)
mock_dialogues_w_api, mock_dialogues_w_api_slot_entries = add_db_api_calls(mock_dialogues_data, mock_slots_dialogues_data)

act2mock_turns = dict()
act2mock_slots = dict()
for dialogue, slot_info in zip(mock_dialogues_w_api, mock_dialogues_w_api_slot_entries):
    for turn_idx, turn in enumerate(dialogue):
        if 'act' in turn.keys():
            # will end up storing the last template(slots_info) applied for act
            act2mock_turns[turn['act']] = turn.copy()
            act2mock_slots[turn['act']] = slot_info[turn_idx].copy()

# create the dataset applying the mappings collected above
mock_templated_dialogues_w_api_calls = []
mock_templated_slots_w_api_calls = []
for dialogue, slots_info in zip(mock_dialogues_w_api, mock_dialogues_w_api_slot_entries):
    mock_templated_dialogue = []
    mock_templated_dialogue_slots = []
    for turn_idx, turn in enumerate(dialogue):
        if 'act' in turn.keys():
            mock_templated_dialogue.append(act2mock_turns[turn['act']])
            mock_templated_dialogue_slots.append(act2mock_slots[turn['act']])
        else:
            mock_templated_dialogue.append(turn.copy())
            mock_templated_dialogue_slots.append(slots_info[turn_idx])
    mock_templated_dialogues_w_api_calls.append(mock_templated_dialogue)
    mock_templated_slots_w_api_calls.append(mock_templated_dialogue_slots)


good_slotfill_dict, templates = extract_slotfill_and_templates(mock_templated_dialogues_w_api_calls, mock_templated_slots_w_api_calls)
unique_slot_names = list(good_slotfill_dict.keys())
initialize_slotfill_model_config_w_paths(evaluate=False)

database = configure_db(mock_templated_dialogues_w_api_calls)
db_pkeys = database.primary_keys

mock_templated_dialogues_w_api_train, mock_templated_dialogues_w_api_test, mock_templated_dialogues_w_api_dev = \
    train_test_val_split(mock_templated_dialogues_w_api_calls)

train_test_val_write(mock_templated_dialogues_w_api_train, mock_templated_dialogues_w_api_test,
                     mock_templated_dialogues_w_api_dev)

gobot_config = configure_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)

