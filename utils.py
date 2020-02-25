import glob
import json
import os
import random
from collections import OrderedDict as odict, defaultdict as dd

from deeppavlov import configs, evaluate_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from sklearn.model_selection import train_test_split

from constants import DSTC8_REPO_PATH, SPEAKER_2_IDX, TEMPLATES_FPATH, DB_FPATH, SLOTFILL_DATA_FPATH, \
    SLOTFILL_CONFIG_FPATH, DATA_DIR, BOT_DATA_DIR, RES_TRN_FPATH, RES_TST_FPATH, RES_VAL_FPATH, TRN_F_AFFIX, \
    TST_F_AFFIX, DEV_F_AFFIX

"""## `def`s"""


# region definitions

def get_dialogue_files_list(dstc8_data_dir_path=DSTC8_REPO_PATH,
                            skip_train=False, skip_test=True, skip_val=True):
    data_dirs = set()
    if not skip_train:
        data_dirs.add(TRN_F_AFFIX)
    if not skip_test:
        data_dirs.add(TST_F_AFFIX)
    if not skip_val:
        data_dirs.add(DEV_F_AFFIX)

    dialogue_files_list = []
    for data_dir in data_dirs:
        dir_files = os.listdir(os.path.join(dstc8_data_dir_path, data_dir))
        dialogue_files_list.extend([(data_dir, os.path.join(dstc8_data_dir_path, data_dir, fn))
                                    for fn in dir_files
                                    if 'schema' not in fn])
    return dialogue_files_list


def mock_slot_value(slot_name):
    return "MOCK_" + slot_name


def template_slot_name(slot_name):
    return '#' + slot_name


def snake_case2camel_case(smth: str):
    res = ''
    for substr in smth.split('_'):
        snake_substr = substr
        camel_substr = snake_substr.capitalize()
        res += camel_substr
    return res


def process_system_turn(turn):
    actions = turn['frames'][0]['actions']
    act = '+'.join(action['act'] + ('_' + snake_case2camel_case(action['slot']) if action['slot'] else '')
                   for action in actions)

    slots = []
    text = turn["utterance"]
    for action in actions:
        if not action['slot']:
            continue

        slot_name = action['slot']
        slot_values = tuple(action['values'])

        if action['act'] != "REQUEST":
            slot_value = '&'.join(slot_values)
            slots.append([slot_name, slot_value])

    fancy_t = odict()
    fancy_t["speaker"] = SPEAKER_2_IDX[turn["speaker"]]
    fancy_t["text"] = text
    fancy_t["slots"] = slots
    fancy_t["act"] = act
    return fancy_t


def process_user_turn(turn):
    frame = turn['frames'][0]
    state = frame['state']

    requested_slots = state["requested_slots"]
    pretty_requested_slots = [['slot', requested_slot] for requested_slot in requested_slots]
    slots = []
    text = turn["utterance"]
    for slot_name, slot_values in state["slot_values"].items():
        slot_value = '&'.join(slot_values)
        slots.append([slot_name, slot_value])

    fancy_t = odict()
    fancy_t["speaker"] = SPEAKER_2_IDX[turn["speaker"]]
    fancy_t["text"] = text
    fancy_t["slots"] = pretty_requested_slots + slots

    return fancy_t


def convert(dstc8_dialogues_data_li):
    mock_li = []

    for f_split, d in dstc8_dialogues_data_li:
        nested_li = []
        for t in d['turns']:
            if 'actions' in t['frames'][0].keys():
                fancy_t = process_system_turn(t)
            else:
                fancy_t = process_user_turn(t)
            fancy_t['data_split'] = f_split
            nested_li.append(fancy_t)
        mock_li.append(nested_li)
    return mock_li


def extract_slot_entries(dstc8_dialogues):
    dialogues_slot_entries = []
    for dialogue in dstc8_dialogues:
        dialogue_slot_entries = []
        for turn in dialogue['turns']:
            frame = turn['frames'][0]
            turn_slot_entries = frame['slots']
            dialogue_slot_entries.append(turn_slot_entries)
        dialogues_slot_entries.append(dialogue_slot_entries)
    return dialogues_slot_entries


def mockify_text(text, slots_info):
    if not slots_info:
        return text, slots_info

    last_span_end = 0
    res_text = ""
    mockified_slots_info = []
    for slot_info in slots_info:
        slot_start = slot_info['start']
        slot_end = slot_info['exclusive_end']
        slot_name = slot_info['slot']

        res_text += text[last_span_end:slot_start]
        mockified_slot_start = len(res_text)
        res_text += mock_slot_value(slot_name)
        mockified_slot_end = len(res_text)

        mockified_slot_info = {
            "start": mockified_slot_start,
            "exclusive_end": mockified_slot_end,
            "slot": slot_name
        }
        mockified_slots_info.append(mockified_slot_info)

        last_span_end = slot_end

    return res_text, mockified_slots_info


def mockify_dialogues(dialogues_data, slot_entries):
    dialogues_data_mockified = dialogues_data.copy()
    dialogues_mockified_slots = []
    for dialogue, dialogue_slots_info in zip(dialogues_data_mockified, slot_entries):
        dialogue_mockified_slots = []
        for turn, turn_slots_info in zip(dialogue, dialogue_slots_info):
            mockified_text, mockified_slots_info = mockify_text(turn['text'], turn_slots_info)
            turn['text'] = mockified_text
            for slot in turn['slots']:
                slot[1] = mock_slot_value(slot[0])
            dialogue_mockified_slots.append(mockified_slots_info)
        dialogues_mockified_slots.append(dialogue_mockified_slots)
    return dialogues_data_mockified, dialogues_mockified_slots


def extract_template(text, slots_info):
    if not slots_info:
        return text

    last_span_end = 0
    res_text = ""
    for slot_info in slots_info:
        slot_start = slot_info['start']
        slot_end = slot_info['exclusive_end']
        slot_name = slot_info['slot']

        res_text += text[last_span_end:slot_start]
        res_text += template_slot_name(slot_name)

        last_span_end = slot_end

    return res_text


def extract_templates(dialogues_data, slot_entries, templates_fpath=TEMPLATES_FPATH):
    templates = dd(set)
    for dialogue, dialogue_slots_info in zip(dialogues_data, slot_entries):
        for turn, turn_slots_info in zip(dialogue, dialogue_slots_info):
            if turn['speaker'] == SPEAKER_2_IDX['SYSTEM']:
                act = turn['act']
                text = turn['text']
                template = extract_template(text, turn_slots_info)
                templates[act].add(template)

    with open(templates_fpath, 'w') as templates_f:
        for k, vals in templates.items():
            for v in vals:
                print('\t'.join((k, v.replace('\n', '`NEWLINE'))), file=templates_f)

    return templates


def fake_api_call_turn(turn, turn_idx, dialogue, mock_slot_name):
    fancy_t = odict()
    fancy_t["speaker"] = 2
    fancy_t["text"] = f"api_call {mock_slot_name}=\"{mock_slot_value(mock_slot_name)}\""
    fancy_t["slots"] = [[mock_slot_name, mock_slot_value(mock_slot_name)]]
    fancy_t["db_result"] = {mock_slot_name: mock_slot_value(mock_slot_name)}
    fancy_t["act"] = "api_call"
    return fancy_t


def add_api_calls(dialogues_data_li, slots_data_li):
    slot_names = {slot_info[0]
                  for dialogue in dialogues_data_li
                  for turn in dialogue
                  for slot_info in turn['slots']}
    random_mock_slot_name = list(slot_names)[0]

    mock_dialogues_w_api_calls = []
    mock_dialogues_w_api_calls_slots_data = []
    for dialogue, slots_info in zip(dialogues_data_li, slots_data_li):
        dialogue_w_api_calls = []
        dialogue_w_api_calls_slots_data = []
        for turn_idx, turn in enumerate(dialogue):
            if any(act.startswith("INFORM") or act.startswith("NOTIFY") for act in turn.get('act', '').split('+')):
                dialogue_w_api_calls.append(fake_api_call_turn(turn, turn_idx, dialogue, random_mock_slot_name))
                dialogue_w_api_calls_slots_data.append([])
            dialogue_w_api_calls.append(turn)
            dialogue_w_api_calls_slots_data.append(slots_info[turn_idx])
        mock_dialogues_w_api_calls.append(dialogue_w_api_calls)
        mock_dialogues_w_api_calls_slots_data.append(dialogue_w_api_calls_slots_data)
    return mock_dialogues_w_api_calls, mock_dialogues_w_api_calls_slots_data


def train_test_val_split(smth_iterable):

    all_the_split_affixes_present = {turn['data_split']
                                     for dialogue in smth_iterable
                                     for turn in dialogue if 'data_split' in turn.keys()}

    if all(affix in all_the_split_affixes_present for affix in {TRN_F_AFFIX, TST_F_AFFIX, DEV_F_AFFIX}):
        train_pt, test_pt, val_pt = list(), list(), list()
        for dialogue in smth_iterable:
            dialogue_data_split = dialogue[0]['data_split']
            if dialogue_data_split == TRN_F_AFFIX:
                train_pt.append(dialogue)
            if dialogue_data_split == TST_F_AFFIX:
                test_pt.append(dialogue)
            if dialogue_data_split == DEV_F_AFFIX:
                val_pt.append(dialogue)
    else:
        train_pt, test_pt = train_test_split(smth_iterable, test_size=0.2)
        train_pt, val_pt = train_test_split(train_pt, test_size=0.2)

    return train_pt, test_pt, val_pt


def configure_db(mock_dialogues_w_api_calls, db_filepath=DB_FPATH):
    all_the_db_responses = [turn['db_result']
                            for dialogue in mock_dialogues_w_api_calls
                            for turn in dialogue
                            if turn.get('db_result')]

    all_the_db_keys = {result_key for result in all_the_db_responses for result_key in result.keys()}

    possible_pkeys = set()
    for pkey_candidate in all_the_db_keys:
        if all(pkey_candidate in result.keys() for result in all_the_db_responses):
            possible_pkeys.add(pkey_candidate)

    db_pkey = random.choice(tuple(possible_pkeys))
    database = Sqlite3Database(primary_keys=[db_pkey], save_path=db_filepath)

    if all_the_db_responses:
        database.fit(all_the_db_responses)

    return database


def extract_slots_info(mock_dialogues_w_api_calls, slotfill_data_fpath=SLOTFILL_DATA_FPATH):
    slotfill_dict = dd(lambda: dd(list))
    for dialogue in mock_dialogues_w_api_calls:
        for turn in dialogue:
            for slot_name, slot_values in turn['slots']:
                for slot_value in slot_values.split('&'):
                    slotfill_dict[slot_name][slot_value] = [slot_value]

    # good means long enough to not to brake the NN
    good_slotfill_dict = dd(lambda: dd(list))
    for slot_name in slotfill_dict.keys():
        if not all(len(slot_value) <= 1 for slot_value in slotfill_dict[slot_name].keys()):
            for slot_value in slotfill_dict[slot_name]:
                if len(slot_value) > 1:
                    good_slotfill_dict[slot_name][slot_value] = [slot_value]

    with open(slotfill_data_fpath, 'w') as slotfill_f:
        json.dump(good_slotfill_dict, slotfill_f, indent=2)

    return good_slotfill_dict


def initialize_slotfill_config_w_paths(res_config_path=SLOTFILL_CONFIG_FPATH,
                                       data_path=DATA_DIR,
                                       slot_vals_path=SLOTFILL_DATA_FPATH,
                                       evaluate=False):
    slotfill_config = read_json(configs.ner.slotfill_simple_dstc2_raw)
    slotfill_config['metadata']['variables']['DATA_PATH'] = data_path
    slotfill_config['metadata']['variables']['SLOT_VALS_PATH'] = slot_vals_path

    if evaluate:
        evaluate_model(slotfill_config)

    json.dump(slotfill_config, open(res_config_path, 'w'))


def setup_gobot_config(db_pkeys, unique_slot_names):
    gobot_config = read_json(configs.go_bot.gobot_simple_dstc2)
    gobot_config['chainer']['pipe'][-1]['embedder'] = None
    gobot_config['chainer']['pipe'][-1]['database'] = {
        'class_name': 'sqlite_database',
        'primary_keys': db_pkeys,
        'save_path': DB_FPATH
    }

    gobot_config['chainer']['pipe'][-1]['slot_filler']['config_path'] = SLOTFILL_CONFIG_FPATH
    gobot_config['chainer']['pipe'][-1]['tracker']['slot_names'] = unique_slot_names
    gobot_config['chainer']['pipe'][-1]['template_type'] = 'DefaultTemplate'
    gobot_config['chainer']['pipe'][-1]['template_path'] = TEMPLATES_FPATH
    gobot_config['metadata']['variables']['DATA_PATH'] = DATA_DIR
    gobot_config['metadata']['variables']['MODEL_PATH'] = BOT_DATA_DIR

    gobot_config['train']['batch_size'] = 8  # batch size
    gobot_config['train']['max_batches'] = 250  # maximum number of training batches
    gobot_config['train']['log_on_k_batches'] = 20
    gobot_config['train']['val_every_n_batches'] = 40  # evaluate on full 'valid' split each n batches
    gobot_config['train']['log_every_n_batches'] = 40  # evaluate on 20 batches of 'train' split every n batches

    return gobot_config


def clear_data_dirs():
    global fpath
    for fpath in glob.glob(os.path.join(DATA_DIR, "*")):
        os.remove(fpath)
    for fpath in glob.glob(os.path.join(BOT_DATA_DIR, "*")):
        os.remove(fpath)


def train_test_val_write(train_data_json, test_data_json, val_data_json,
                         trn_fpath=RES_TRN_FPATH, tst_fpath=RES_TST_FPATH, val_fpath=RES_VAL_FPATH):
    with open(trn_fpath, 'w') as fpath:
        json.dump(train_data_json, fpath, indent=2)

    with open(tst_fpath, 'w') as fpath:
        json.dump(test_data_json, fpath, indent=2)

    with open(val_fpath, 'w') as fpath:
        json.dump(val_data_json, fpath, indent=2)


def base_conversion(dstc_dialogues_li):
    slot_entries = extract_slot_entries([dialogue for f_split, dialogue in dstc_dialogues_li])
    dialogues_data = convert(dstc_dialogues_li)
    return dialogues_data, slot_entries


def extract_metainfo(dstc2_base_dialogues, dstc2_base_dialogues_slot_entries):
    slotfill_data = extract_slots_info(dstc2_base_dialogues)
    templates = extract_templates(dstc2_base_dialogues, dstc2_base_dialogues_slot_entries)
    return slotfill_data, templates

# endregion definitions
