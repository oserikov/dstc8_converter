import glob
import json
import os
import random
from collections import OrderedDict as odict, defaultdict as dd

from deeppavlov import configs, evaluate_model
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from sklearn.model_selection import train_test_split

from constants import *


def clear_data_dirs():
    for fpath in glob.glob(os.path.join(DATA_DIR, "*")):
        os.remove(fpath)
    for fpath in glob.glob(os.path.join(BOT_DATA_DIR, "*")):
        os.remove(fpath)


def get_dstc8_dialogue_files_list(dstc8_data_dir_path=DSTC8_REPO_PATH,
                                  skip_train=False, skip_test=True, skip_val=True):
    data_parts_names = set()
    if not skip_train:
        data_parts_names.add(TRN_F_AFFIX)
    if not skip_test:
        data_parts_names.add(TST_F_AFFIX)
    if not skip_val:
        data_parts_names.add(DEV_F_AFFIX)

    dialogues_files_list = []
    for data_part_name in data_parts_names:
        data_part_filenames = os.listdir(os.path.join(dstc8_data_dir_path, data_part_name))
        dialogues_files_list.extend([(data_part_name, os.path.join(dstc8_data_dir_path, data_part_name, fn))
                                    for fn in data_part_filenames
                                    if 'schema' not in fn])
    return dialogues_files_list


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

# region dstc8 to dstc2 conversion

def system_dstc8_2_dstc2_turn(dstc8_turn):
    actions = dstc8_turn['frames'][0]['actions']

    # act: Request; slot: phone -> Request_phone
    act = '+'.join(action['act'] + ('_' + snake_case2camel_case(action['slot'])
                                    if action['slot'] else '')
                   for action in actions)

    slots = []
    for action in actions:
        if not action['slot']:
            continue

        slot_name = action['slot']
        slot_values = tuple(action['values'])

        if action['act'] != "REQUEST":
            slot_value = '&'.join(slot_values)
            slots.append([slot_name, slot_value])

    simple_dstc2_turn = odict()
    simple_dstc2_turn["speaker"] = SPEAKER_2_IDX[dstc8_turn["speaker"]]
    simple_dstc2_turn["text"] = dstc8_turn["utterance"]
    simple_dstc2_turn["slots"] = slots
    simple_dstc2_turn["act"] = act
    return simple_dstc2_turn


def user_dstc8_2_dstc2_turn(dstc8_turn):
    frame = dstc8_turn['frames'][0]
    state = frame['state']

    requested_slots = state["requested_slots"]
    pretty_requested_slots = [['slot', requested_slot] for requested_slot in requested_slots]
    slots = []
    for slot_name, slot_values in state["slot_values"].items():
        slot_value = '&'.join(slot_values)
        slots.append([slot_name, slot_value])

    simple_dstc2_turn = odict()
    simple_dstc2_turn["speaker"] = SPEAKER_2_IDX[dstc8_turn["speaker"]]
    simple_dstc2_turn["text"] = dstc8_turn["utterance"]
    simple_dstc2_turn["slots"] = pretty_requested_slots + slots
    
    actions = dstc8_turn['frames'][0]['actions']

    # act: Request; slot: phone -> Request_phone
    act = '+'.join(action['act'] + ('_' + snake_case2camel_case(action['slot'])
                                    if action['slot'] else '')
                   for action in actions)
    simple_dstc2_turn["act"] = act
    return simple_dstc2_turn


def dstc8_dialogues_2_dstc2_dialogues(dstc8_dialogues):
    dstc2_dialogues = []

    for data_part_name, dialogue in dstc8_dialogues:
        dstc2_dialogue = []
        for dstc8_turn in dialogue['turns']:
            if dstc8_turn["speaker"] == "SYSTEM":
                dstc2_turn = system_dstc8_2_dstc2_turn(dstc8_turn)
            else:
                dstc2_turn = user_dstc8_2_dstc2_turn(dstc8_turn)
            dstc2_turn['data_split'] = data_part_name
            dstc2_dialogue.append(dstc2_turn)
        dstc2_dialogues.append(dstc2_dialogue)
    return dstc2_dialogues

def extract_slots_spans(dstc8_dialogues):
    dialogues_slots_spans = []
    for dialogue in dstc8_dialogues:
        dialogue_slots_spans = []
        for turn in dialogue['turns']:
            frame = turn['frames'][0]
            turn_slots_span = frame['slots']
            dialogue_slots_spans.append(turn_slots_span)
        dialogues_slots_spans.append(dialogue_slots_spans)
    return dialogues_slots_spans

def base_dstc8_2_dstc2(dstc_dialogues_li):
    slot_spans = extract_slots_spans([dialogue for f_split, dialogue in dstc_dialogues_li])
    dialogues_data = dstc8_dialogues_2_dstc2_dialogues(dstc_dialogues_li)
    return dialogues_data, slot_spans

# endregion dstc8 to dstc2 conversion

# region datasets modification

def mockify_slots_in_text(text, text_slots_spans):
    if not text_slots_spans:
        return text, text_slots_spans

    res_text = ""
    last_processed_slot_end = 0
    mockified_slots_spans = []
    for slot_span in text_slots_spans:
        slot_start = slot_span['start']
        slot_end = slot_span['exclusive_end']
        slot_name = slot_span['slot']

        res_text += text[last_processed_slot_end:slot_start]
        mockified_slot_start = len(res_text)
        res_text += mock_slot_value(slot_name)
        mockified_slot_end = len(res_text)

        mockified_slot_span = {
            "start": mockified_slot_start,
            "exclusive_end": mockified_slot_end,
            "slot": slot_name
        }
        mockified_slots_spans.append(mockified_slot_span)

        last_processed_slot_end = slot_end

    return res_text, mockified_slots_spans


def mockify_slots_in_dialogues(dstc2_dialogues, dialogues_slots_spans):
    dialogues_mockified = dstc2_dialogues.copy()
    dialogues_mockified_slots_spans = []

    for dialogue, dialogue_slots_spans in zip(dialogues_mockified, dialogues_slots_spans):
        dialogue_mockified_slots = []

        for turn, turn_slots_spans in zip(dialogue, dialogue_slots_spans):
            mockified_text, mockified_slots_spans = mockify_slots_in_text(turn['text'], turn_slots_spans)

            turn['text'] = mockified_text
            for slot_info in turn['slots']:
                slot_info[1] = mock_slot_value(slot_info[0])

            dialogue_mockified_slots.append(mockified_slots_spans)
        dialogues_mockified_slots_spans.append(dialogue_mockified_slots)
    return dialogues_mockified, dialogues_mockified_slots_spans


def fake_api_call_turn(turn, turn_idx, dialogue, mock_slot_name):
    # unused params passed to allow sophisticated fake api generation if needed

    dstc2_api_call_turn = odict()
    dstc2_api_call_turn["speaker"] = SPEAKER_2_IDX["SYSTEM"]
    dstc2_api_call_turn["text"] = f"api_call {mock_slot_name}=\"{mock_slot_value(mock_slot_name)}\""
    dstc2_api_call_turn["slots"] = [[mock_slot_name, mock_slot_value(mock_slot_name)]]
    dstc2_api_call_turn["db_result"] = {mock_slot_name: mock_slot_value(mock_slot_name)}
    dstc2_api_call_turn["act"] = "api_call"
    return dstc2_api_call_turn


def add_db_api_calls(dstc2_dialogues, dialogues_slots_spans):
    slot_names = {slot_info[0]
                  for dialogue in dstc2_dialogues
                  for turn in dialogue
                  for slot_info in turn['slots']}
    mock_slot_name = list(slot_names)[0]

    dialogues_w_api_calls = []
    dialogues_w_api_calls_slots_spans = []
    for dialogue, slots_spans in zip(dstc2_dialogues, dialogues_slots_spans):
        dialogue_w_api_calls = []
        dialogue_w_api_calls_slots_spans = []
        for turn_idx, turn in enumerate(dialogue):
            if any(act.startswith("INFORM") or act.startswith("NOTIFY")
                   for act in turn.get('act', '').split('+')):
                dialogue_w_api_calls.append(fake_api_call_turn(turn, turn_idx, dialogue, mock_slot_name))
                dialogue_w_api_calls_slots_spans.append([])
            dialogue_w_api_calls.append(turn)
            dialogue_w_api_calls_slots_spans.append(slots_spans[turn_idx])
        dialogues_w_api_calls.append(dialogue_w_api_calls)
        dialogues_w_api_calls_slots_spans.append(dialogue_w_api_calls_slots_spans)
    return dialogues_w_api_calls, dialogues_w_api_calls_slots_spans

def reduce_action_replics_variance(dialogues, slot_spans):
    act2mock_turns = dict()
    act2mock_slots = dict()
    for dialogue, slot_info in zip(dialogues, slot_spans):
        for turn_idx, turn in enumerate(dialogue):
            if 'act' in turn.keys():
                # will end up storing the last template(slots_info) applied for act
                act2mock_turns[turn['act']] = turn.copy()
                act2mock_slots[turn['act']] = slot_info[turn_idx].copy()

    # create the dataset applying the mappings collected above
    dialogues_modified = []
    slot_spans_modified = []
    for dialogue, slot_spans in zip(dialogues, slot_spans):
        dialogue_modified = []
        dialogue_slot_spans_modified = []
        for turn_idx, turn in enumerate(dialogue):
            if 'act' in turn.keys():
                dialogue_modified.append(act2mock_turns[turn['act']])
                dialogue_slot_spans_modified.append(act2mock_slots[turn['act']])
            else:
                dialogue_modified.append(turn.copy())
                dialogue_slot_spans_modified.append(slot_spans[turn_idx])
        dialogues_modified.append(dialogue_modified)
        slot_spans_modified.append(dialogue_slot_spans_modified)

    return dialogues_modified, slot_spans_modified
# endregion datasets modification

# region metainfo extraction

def extract_slotfill_data_config(dialogues, slotfill_data_fpath=SLOTFILL_DATA_FPATH):
    draft_slotfill_data_config = dd(lambda: dd(list))
    for dialogue in dialogues:
        for turn in dialogue:
            for slot_name, slot_values in turn['slots']:
                for slot_value in slot_values.split('&'):
                    draft_slotfill_data_config[slot_name][slot_value] = [slot_value]

    # we should remove slot values of length 1
    # cause levenshtein pipeline fails on them
    slotfill_data_config = dd(lambda: dd(list))
    for slot_name in draft_slotfill_data_config.keys():
        if not all(len(slot_value) <= 1
                   for slot_value in draft_slotfill_data_config[slot_name].keys()):

            for slot_value in draft_slotfill_data_config[slot_name]:
                if len(slot_value) > 1:
                    slotfill_data_config[slot_name][slot_value] = [slot_value]

    with open(slotfill_data_fpath, 'w') as slotfill_f:
        json.dump(slotfill_data_config, slotfill_f, indent=2)

    return slotfill_data_config

def text2template_text(text, text_slots_spans):
    if not text_slots_spans:
        return text

    tepmlate_text = ""
    last_processed_slot_end = 0
    for slot_span in text_slots_spans:
        slot_start = slot_span['start']
        slot_end = slot_span['exclusive_end']
        slot_name = slot_span['slot']

        tepmlate_text += text[last_processed_slot_end:slot_start]
        tepmlate_text += template_slot_name(slot_name)

        last_processed_slot_end = slot_end

    return tepmlate_text


def extract_action_templates(dstc2_dialogues, dialogues_slots_spans, templates_fpath=TEMPLATES_FPATH):
    action_templates = dd(set)
    for dialogue, dialogue_slots_span in zip(dstc2_dialogues, dialogues_slots_spans):
        for turn, turn_slots_info in zip(dialogue, dialogue_slots_span):
            if turn['speaker'] == SPEAKER_2_IDX['SYSTEM']:
                act = turn['act']
                text = turn['text']
                template_text = text2template_text(text, turn_slots_info)
                action_templates[act].add(template_text)

    with open(templates_fpath, 'w') as templates_f:
        for action, templates in action_templates.items():
            for template in templates:
                print('\t'.join((action, template.replace('\n', '`NEWLINE'))), file=templates_f)

    return action_templates


def extract_slotfill_and_templates(dstc2_base_dialogues, dstc2_base_dialogues_slot_entries):
    slotfill_data = extract_slotfill_data_config(dstc2_base_dialogues)
    templates = extract_action_templates(dstc2_base_dialogues, dstc2_base_dialogues_slot_entries)
    return slotfill_data, templates


def configure_db(dialogues_w_api_calls, db_filepath=DB_FPATH):
    db_responses = [turn['db_result']
                    for dialogue in dialogues_w_api_calls
                    for turn in dialogue
                    if turn.get('db_result')]

    all_the_db_keys = {result_key
                       for result in db_responses
                       for result_key in result.keys()}

    possible_pkeys = set()
    for pkey_candidate in all_the_db_keys:
        if all(pkey_candidate in result.keys()
               for result in db_responses):
            possible_pkeys.add(pkey_candidate)

    db_pkey = random.choice(tuple(possible_pkeys))
    database = Sqlite3Database(primary_keys=[db_pkey], save_path=db_filepath)

    if db_responses:
        database.fit(db_responses)

    return database

# endregion metainfo extraction

# region train test val tools

def train_test_val_split(dialogues):

    all_the_data_parts_present = {turn['data_split']
                                  for dialogue in dialogues
                                  for turn in dialogue if 'data_split' in turn.keys()}

    if all(affix in all_the_data_parts_present
           for affix in {TRN_F_AFFIX, TST_F_AFFIX, DEV_F_AFFIX}):

        train_dialogues, test_dialogues, val_dialogues = list(), list(), list()
        for dialogue in dialogues:
            dialogue_data_split = dialogue[0]['data_split']
            if dialogue_data_split == TRN_F_AFFIX:
                train_dialogues.append(dialogue)
            if dialogue_data_split == TST_F_AFFIX:
                test_dialogues.append(dialogue)
            if dialogue_data_split == DEV_F_AFFIX:
                val_dialogues.append(dialogue)
    else:
        train_dialogues, test_dialogues = train_test_split(dialogues, test_size=0.2)
        train_dialogues, val_dialogues = train_test_split(train_dialogues, test_size=0.2)

    return train_dialogues, test_dialogues, val_dialogues


def train_test_val_write(train_data_dict, test_data_dict, val_data_dict,
                         trn_fpath=RES_TRN_FPATH, tst_fpath=RES_TST_FPATH, val_fpath=RES_VAL_FPATH):
    with open(trn_fpath, 'w') as fpath:
        json.dump(train_data_dict, fpath, indent=2)

    with open(tst_fpath, 'w') as fpath:
        json.dump(test_data_dict, fpath, indent=2)

    with open(val_fpath, 'w') as fpath:
        json.dump(val_data_dict, fpath, indent=2)

# endregion train test val tools

# region configs utils

def initialize_slotfill_model_config_w_paths(res_config_path=SLOTFILL_CONFIG_FPATH,
                                             data_path=DATA_DIR,
                                             slot_vals_path=SLOTFILL_DATA_FPATH,
                                             evaluate=False):

    slotfill_config = read_json(configs.ner.slotfill_simple_dstc2_raw)
    slotfill_config['metadata']['variables']['DATA_PATH'] = data_path
    slotfill_config['metadata']['variables']['SLOT_VALS_PATH'] = slot_vals_path

    if evaluate:
        evaluate_model(slotfill_config)

    json.dump(slotfill_config, open(res_config_path, 'w'))


def configure_gobot_config(db_pkeys, unique_slot_names):
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

# endregion configs utils
