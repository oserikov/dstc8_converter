from deeppavlov import train_model
from utils import *
from constants import DOMAIN_OF_INTEREST, DATA_DIR, BOT_DATA_DIR

clear_data_dirs()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BOT_DATA_DIR, exist_ok=True)

dstc8_dialogues_filenames = get_dstc8_dialogue_files_list()
dstc8_dialogues = [(f_split, json.load(open(fn))) for f_split, fn in dstc8_dialogues_filenames]
relevant_dstc8_dialogues = [(f_split, dialogue)
                            for f_split, dialogues_file in dstc8_dialogues
                            for dialogue in dialogues_file
                            if dialogue['services'] == [DOMAIN_OF_INTEREST]]


# create the dataset applying the mappings collected above
dstc2_dialogues, slot_spans = reduce_action_replics_variance(
                                  * add_db_api_calls(
                                      * mockify_slots_in_dialogues(
                                          * base_dstc8_2_dstc2(relevant_dstc8_dialogues))))

action_aggregated_dialogues = dstc2_dialogues.copy()
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


slotfill_data, templates = extract_slotfill_and_templates(action_aggregated_dialogues, slot_spans)
initialize_slotfill_model_config_w_paths(evaluate=False)
unique_slot_names = list(slotfill_data.keys())

database = configure_db(action_aggregated_dialogues)
db_pkeys = database.primary_keys

dialogues_train, dialogues_test, dialogues_dev = train_test_val_split(action_aggregated_dialogues)
train_test_val_write(dialogues_train, dialogues_test, dialogues_dev)

gobot_config = configure_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)
