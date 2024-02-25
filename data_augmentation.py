import numpy as np
import pandas as pd
import re
import random
import itertools
from tensorflow.keras.preprocessing.sequence import pad_sequences

from templates import *


replaces = [
    [" м.", " місто "],
    [",м.", ",місто "],
    [' вул.', " вулиця ", " в."],
    [',вул.', ",вулиця ", ",в."],
    [' буд.', ' буд ', ' будинок '],
    [',буд.', ',буд ', ',будинок '],
    [",кв.", ",квартира "],
    [" кв.", " квартира "],
    ['р-н', 'район'],
    ["бульвар ", "бульв.", "б-р "],
    ["пров.", "пров ", "провулок "],
    ["просп.", "проспект "],
    ["площа ", "пл."],
    [" село ", " с.", " селище "],
    [",село ", ",с.", ",селище "],
    ["обл.", "область "],
    ["обл,", "область,"],
    ["обл ", "область "]]


def add_noise_to_address(address, noise):

    non_alpha_numeric_positions = [m.start() for m in re.finditer(r"[^\w]", address)]
    non_alpha_numeric_positions.append('endl')

    if non_alpha_numeric_positions:
        random_position = random.choice(non_alpha_numeric_positions)
        if random_position == 'endl':
            modified_address = address + noise
        else:
            modified_address = address[:random_position] + noise + address[random_position+1:]
        return modified_address
    else:
        modified_address = address
        return modified_address


def get_domains(address):
    pattern = r"\{(.*?)\}"
    matches = re.findall(pattern, address)
    return matches


def find_replacement_match(address, replacement_set):
    for replacement in replacement_set:
        if replacement in address:
            return replacement


def generate_template_dict(real=True, house_tail_proba=0):

    if real == 'mix':
        tamplate_dict_real = {
            'avenue': np.random.choice(avenues),
            'city': np.random.choice(cities),
            'region': np.random.choice(regions),
            'district': np.random.choice(districts),
            'village': np.random.choice(villages),
            'street': np.random.choice(streets),
            'lane': np.random.choice(lanes),
            'house_num': str(np.random.randint(1, 120)),
            'flat_num': np.random.randint(1, 120),
            'index': ''.join(str(np.random.randint(0, 10)) for _ in range(5))
        }

        tamplate_dict_random = {
            'avenue': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'city': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'region': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'district': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'village': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'street': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'lane': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'house_num': str(np.random.randint(1, 120)),
            'flat_num': np.random.randint(1, 120),
            'index': ''.join(str(np.random.randint(0, 10)) for _ in range(5))
        }

        tamplate_dict = {key: random.choice([tamplate_dict_real[key], tamplate_dict_random[key]]) for key in tamplate_dict_real.keys()}

    elif real:
        tamplate_dict = {
            'avenue': np.random.choice(avenues),
            'city': np.random.choice(cities),
            'region': np.random.choice(regions),
            'district': np.random.choice(districts),
            'village': np.random.choice(villages),
            'street': np.random.choice(streets),
            'lane': np.random.choice(lanes),
            'house_num': str(np.random.randint(1, 120)),
            'flat_num': np.random.randint(1, 120),
            'index': ''.join(str(np.random.randint(0, 10)) for _ in range(5))
        }

    else:
        tamplate_dict = {
            'avenue': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'city': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'region': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'district': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'village': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'street': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'lane': ''.join(random.choices(ukrainian_alphabet + ' ', k=random.randint(4, 14))),
            'house_num': str(np.random.randint(1, 120)),
            'flat_num': np.random.randint(1, 120),
            'index': ''.join(str(np.random.randint(0, 10)) for _ in range(5))
        }

    if house_tail_proba > 0 and np.random.rand() < house_tail_proba:
        tamplate_dict['house_num'] += random.choice(['-а', '-б', '-в', '-г'] + ['к' + str(n) for n in range(1, 8)] + ['/' + str(n) for n in range(1, 8)])

    return tamplate_dict


def generate_address(address_template, template_dict):
    address = address_template.format(**template_dict)
    return address


def add_noise(address, proba):
    if np.random.rand() < proba:
        noise = random.choice(noises)
        address = add_noise_to_address(address, noise.format(noise=random.choice([' ', ' ', ',', 'а', '-', 'к', '1', '2', '3'])))
    return address


new_templates = address_templates.copy()

for replacement_set in replaces:
    current_count = len(new_templates)
    for ind, template in enumerate(new_templates):
        if ind == current_count:
            new_templates = list(set(new_templates))
            break
        replacement_token = find_replacement_match(template, replacement_set)
        if replacement_token:
            for domain_token in replacement_set:
                new_templates.append(template.replace(replacement_token, domain_token))


available_diff = {'index', 'region', 'district'}
addresses_pairs = list(itertools.product(new_templates, new_templates))

true_addresses_pairs = []
for template_1, template_2 in addresses_pairs:
    if not set(get_domains(template_1)).symmetric_difference(set(get_domains(template_2))) - available_diff:
        true_addresses_pairs.append((template_1, template_2))

TRUE_ADDRESS_TEMPLATES_PAIRS = true_addresses_pairs
ALL_ADDRESS_TEMPLATES_PAIRS = addresses_pairs

del new_templates


def generate_address_pair(true_addresses_pairs=TRUE_ADDRESS_TEMPLATES_PAIRS,
                          all_addresses_pairs=ALL_ADDRESS_TEMPLATES_PAIRS,
                          label='random',
                          noise_proba=0.05):
    if label == 'random':
        label = np.random.choice([True, False])

    if label:
        address_template_1, address_template_2 = random.choice(true_addresses_pairs)
        template_dict = generate_template_dict(real='mix', house_tail_proba=0.02)

        address_1 = generate_address(address_template_1, template_dict)
        address_1 = add_noise(address_1, noise_proba)

        address_2 = generate_address(address_template_2, template_dict)
        address_2 = add_noise(address_2, noise_proba)

        return address_1, address_2, label

    else:
        address_template_1, address_template_2 = random.choice(all_addresses_pairs)
        template_dict_1 = generate_template_dict(real='mix', house_tail_proba=0.02)
        template_dict_2 = generate_template_dict(real='mix', house_tail_proba=0.02)

        address_1 = generate_address(address_template_1, template_dict_1)
        address_1 = add_noise(address_1, noise_proba)

        address_2 = generate_address(address_template_2, template_dict_2)
        address_2 = add_noise(address_2, noise_proba)

        return address_1, address_2, label


MAX_LEN = 120
BATCH_SIZE = 32

char_to_index = {char: idx + 1 for idx, char in enumerate(allowed_chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}

def tokenize_address(address):
    return [char_to_index[char] for char in address if char in char_to_index]

def generate_address_batch(batch_size):

    X_1 = []
    X_2 = []
    y = []
    for _ in range(batch_size):
        address_1, address_2, label = generate_address_pair()
        tokenized_adress_1 = tokenize_address(address_1)
        tokenized_adress_2 = tokenize_address(address_2)
        X_1.append(tokenized_adress_1)
        X_2.append(tokenized_adress_2)
        y.append(int(label))

    adress_1_padded = pad_sequences(X_1, maxlen=MAX_LEN)
    adress_2_padded = pad_sequences(X_2, maxlen=MAX_LEN)

    return (adress_1_padded, adress_2_padded), np.array(y, dtype=np.int32)


def data_generator():
    for _ in range(2**19):
        yield generate_address_batch(BATCH_SIZE)


VALID_DATA = generate_address_batch(2**14)


def process_address(address, allowed_chars):

    address = address.lower()
    address = ''.join([char for char in address if char in allowed_chars])
    address = address.strip()

    return address


def df_to_model_input(df, allowed_chars):
    df = df.astype({'adress_1': 'str', 'adress_2': 'str'})
    df['adress_1'] = df['adress_1'].apply(lambda x: process_address(x, allowed_chars))
    df['adress_2'] = df['adress_2'].apply(lambda x: process_address(x, allowed_chars))
    tokenized_adress_1 = df['adress_1'].apply(tokenize_address)
    tokenized_adress_2 = df['adress_2'].apply(tokenize_address)
    adress_1_padded = pad_sequences(tokenized_adress_1, maxlen=MAX_LEN)
    adress_2_padded = pad_sequences(tokenized_adress_2, maxlen=MAX_LEN)
    return adress_1_padded, adress_2_padded


test_df = pd.read_csv('test_data.csv')
test_data = df_to_model_input(test_df, allowed_chars)
test_labels = test_df['is_same'].astype(int)

TEST_DATA = test_data, test_labels
