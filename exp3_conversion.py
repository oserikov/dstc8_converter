from deeppavlov import train_model
from utils import *
from constants import DOMAIN_OF_INTEREST, DATA_DIR, BOT_DATA_DIR


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

action_aggregated_dialogues = mock_templated_dialogues_w_api_calls.copy()
for dialogue in action_aggregated_dialogues:
    for turn in dialogue:
        if "act" in turn.keys():
            turn_acts = turn["act"].split("+")
            res_acts = []
            base_acts = {"REQUEST", "INFORM", "OFFER", "CONFIRM"}
            # keep the non-base actions
            for act in turn_acts:
                if not any(act.startswith(prefix) for prefix in base_acts):
                    res_acts.append(act)

            # aggregate all the base actions entries
            for base_act in base_acts:
                if any(act.startswith(base_act) for act in turn_acts):
                    res_acts.append(base_act + "_needed")
            turn['act'] = '+'.join(res_acts)


good_slotfill_dict, templates = extract_slotfill_and_templates(action_aggregated_dialogues, mock_templated_slots_w_api_calls)
unique_slot_names = list(good_slotfill_dict.keys())
initialize_slotfill_model_config_w_paths(evaluate=False)

database = configure_db(action_aggregated_dialogues)
db_pkeys = database.primary_keys

action_aggregated_dialogues_train, action_aggregated_dialogues_test, action_aggregated_dialogues_dev = \
    train_test_val_split(action_aggregated_dialogues)

train_test_val_write(action_aggregated_dialogues_train, action_aggregated_dialogues_test,
                     action_aggregated_dialogues_dev)

gobot_config = configure_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)

