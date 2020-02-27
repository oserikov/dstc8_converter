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


dstc2_dialogues, slot_spans = add_db_api_calls(* base_dstc8_2_dstc2(relevant_dstc8_dialogues))
slotfill_data, templates = extract_slotfill_and_templates(dstc2_dialogues, slot_spans)
initialize_slotfill_model_config_w_paths(evaluate=False)
unique_slot_names = list(slotfill_data.keys())

database = configure_db(dstc2_dialogues)
db_pkeys = database.primary_keys

dialogues_train, dialogues_test, dialogues_dev = train_test_val_split(dstc2_dialogues)
train_test_val_write(dialogues_train, dialogues_test, dialogues_dev)

# WARNING! will train for long time because of the huge number of possible slot values
gobot_config = configure_gobot_config(db_pkeys, unique_slot_names)
train_model(gobot_config)
