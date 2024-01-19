import os
import sqlite3

import numpy as np
import pandas as pd

from qatch.database_reader import SingleDatabase


def save_spider_format_for_db_id(df, db: SingleDatabase, model_name: str | None = None):
    db_save_path = db.db_path
    test_file_name = f'{model_name}_{db.db_name}_test.json' if model_name is not None else f'{db.db_name}_test.json'
    tables_file_name = f'{model_name}_{db.db_name}_tables.json' if model_name is not None else f'{db.db_name}_tables.json'
    path = os.path.join(db_save_path, test_file_name)
    df.to_json(path, orient='records')

    table_names = db.table_names

    # for each table
    for table_id, name in enumerate(table_names):
        table_schema = db.get_schema_given(table_name=name)
        column_names = [[id_col, col_name]
                        for id_col, col_name in enumerate(table_schema.name.tolist())]
        column_types = table_schema.type.tolist()

        # create SPIDER dataframe for tables.json
        tables = pd.DataFrame({
            'column_names': [column_names],
            'column_names_original': [column_names],
            'column_types': [column_types],
            'db_id': db.db_name,
            'foreign_keys': [[]],
            'primary_keys': [[]],
            'table_names': [table_names],
            'table_names_original': [table_names],
        })

        path = os.path.join(db_save_path, tables_file_name)
        tables.to_json(path, orient='records')
        return df, tables


def read_breast_cancer_dataset(df: pd.DataFrame,
                               sample_size: int | None = None,
                               random_state: int = 2023) -> pd.DataFrame:
    """
    10 columns. TAPAS 45 rows, TAPEX 30 rows, chatGPT 35 rows
    * remove columns 'rfstime' and 'age'
    * normalize columns meno, hormon, status
    * rename columns
    """

    df = df.drop(['Unnamed: 0', 'rfstime'], axis='columns')
    if sample_size is None:
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        df = df.sample(sample_size, random_state=random_state, ignore_index=True)

    id2meno = {0: 'premenopausal', 1: 'postmenopausal'}
    id2hormon = {0: 'no', 1: 'yes'}
    id2status = {0: 'aliveWithoutRecurrence,', 1: 'recurrenceOrDeath'}

    df['meno'] = df.apply(lambda row: id2meno[row['meno']], axis=1)
    df['hormon'] = df.apply(lambda row: id2hormon[row['hormon']], axis=1)
    df['status'] = df.apply(lambda row: id2status[row['status']], axis=1)

    df = df.rename({'pid': 'patientIdentifier', 'meno': 'menopausalStatus',
                    'size': 'tumorSize', 'grade': 'tumorGrade',
                    'nodes': 'numberPositiveLymphNodes',
                    'pgr': 'progesteroneReceptor', 'er': 'estrogenReceptor',
                    'hormon': 'hormonalTherapy'
                    },
                   axis='columns')
    df.columns = df.columns.str.lower()
    return df


def read_heart_attack_dataset(df,
                              sample_size: int | None = None,
                              random_state: int = 2023) -> pd.DataFrame:
    """
    10 columns. TAPAS 45 rows, TAPEX 30 rows, chatGPT 30 rows
    * remove columns "oldpeak", "slp", "thalachh", "exng"
    * normalize columns "cp", "fbs", "restecg", "output", "sex"
    * rename columns
    """
    df = df.drop(["oldpeak", "slp", "thalachh", "exng"], axis='columns')

    if sample_size is None:
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        df = df.sample(sample_size, random_state=random_state, ignore_index=True)

    id2chest_pain = {0: 'typicalAngina',
                     1: 'atypicalAngina',
                     2: 'nonAnginalPain',
                     3: 'asymptomatic'}
    id2fbs = {0: 'false', 1: 'true'}
    id2rest_ecg = {0: 'normal', 1: 'STTWaveAbnormality',
                   2: 'leftVentricularHypertrophy'}
    id2target = {0: 'noHeartAttack', 1: 'heartAttack'}
    id2sex = {0: 'male', 1: 'female'}

    df['sex'] = df.apply(lambda row: id2sex[row.sex], axis=1)
    df['cp'] = df.apply(lambda row: id2chest_pain[row['cp']], axis=1)
    df['fbs'] = df.apply(lambda row: id2fbs[row.fbs], axis=1)
    df['restecg'] = df.apply(lambda row: id2rest_ecg[row.restecg], axis=1)
    df['output'] = df.apply(lambda row: id2target[row.output], axis=1)

    df = df.rename({'cp': 'ChestPaintype',
                    'trtbps': 'restingBloodPressure',
                    'chol': 'cholestoralInMg',
                    'fbs': 'fastingBloodSugar',
                    'restecg': 'restingElectrocardiographicRresults',
                    'caa': 'numberOfMajorVvessels'}, axis='columns')
    df.columns = df.columns.str.lower()

    return df


def read_bank_fraud_dataset(df,
                            sample_size: int | None = None,
                            random_state: int = 2023) -> pd.DataFrame:
    """
    10 columns. TAPAS 45 rows, TAPEX _ rows, chatGPT 30 rows
    * manually select 10 columns
    """
    df = df.loc[:, ['has_other_cards',
                    'housing_status',  # Anonymized
                    'date_of_birth_distinct_emails_4w',
                    'income',
                    'payment_type',  # Anonymized
                    'employment_status',  # Anonymized
                    'credit_risk_score',
                    'session_length_in_minutes',
                    'device_os',
                    'email_is_free']]
    if sample_size is None:
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        df = df.sample(sample_size, random_state=random_state, ignore_index=True)
    df['has_other_cards'] = df.has_other_cards.map(lambda x: 'True' if x == 1 else 'False')
    df['email_is_free'] = df.email_is_free.map(lambda x: 'Free' if x == 0 else 'Paid')
    df = df.rename({'has_other_cards': 'hasOtherCards',
                    'housing_status': 'housingStatus',
                    'date_of_birth_distinct_emails_4w': 'dateOfBirthDistinctEmails4w',
                    'payment_type': 'paymentType',
                    'employment_status': 'employmentStatus',
                    'credit_risk_score': 'creditRiskScore',
                    'session_length_in_minutes': 'sessionLengthMinutes',
                    'device_os': 'deviceOs',
                    'email_is_free': 'emailIsFree',
                    }, axis='columns')
    df.columns = df.columns.str.lower()

    return df


def read_finance_factory_ibm(df,
                             sample_size: int | None = None,
                             random_state: int = 2023) -> pd.DataFrame:
    """10 columns. TAPAS 45 rows, TAPEX 20 rows, chatGPT 30 rows"""
    df = df.drop(['countryCode', 'SettledDate'], axis='columns')
    df.columns = df.columns.str.lower()
    if sample_size is None:
        return df.sample(frac=1, random_state=random_state, ignore_index=True)
    return df.sample(sample_size, random_state=random_state, ignore_index=True)


def read_fitness_trackers_dataset(df,
                                  sample_size: int | None = None,
                                  random_state: int = 2023) -> pd.DataFrame:
    """10 columns. TAPAS 50 rows, TAPEX 25 rows, chatGPT 30 rows"""
    df = df.drop('Reviews', axis='columns')
    df = df.dropna()
    df = df.rename({'Brand Name': 'brandname',
                    'Device Type': 'devicetype',
                    'Model Name': 'modelname',
                    'Selling Price': 'sellingprice',
                    'Original Price': 'originalprice',
                    'Rating (Out of 5)': 'rating',
                    'Strap Material': 'strapmaterial',
                    'Average Battery Life (in days)': 'averagebatterylife',
                    }, axis='columns')
    df.columns = df.columns.str.lower()
    if sample_size is None:
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        df = df.sample(sample_size, random_state=random_state, ignore_index=True)

    df['sellingprice'] = df['sellingprice'].map(lambda x: float(x.replace(',', '')))
    df['originalprice'] = df['originalprice'].map(lambda x: float(x.replace(',', '')))

    return df


def read_sales_transactions_dataset(df,
                                    sample_size: int | None = None,
                                    random_state: int = 2023) -> pd.DataFrame:
    """8 columns. TAPAS 60 rows, TAPEX 25 rows, chatGPT 40 """
    df['ProductName'] = df.ProductName.map(lambda x: "-".join(x.split()))
    df['Country'] = df.Country.map(lambda x: "-".join(x.split()))
    df.columns = df.columns.str.lower()
    if sample_size is None:
        return df.sample(frac=1, random_state=random_state, ignore_index=True)
    return df.sample(sample_size, random_state=random_state, ignore_index=True)


def read_adult_dataset(df,
                       sample_size: int | None = None,
                       random_state: int = 2023) -> pd.DataFrame:
    """TAPAS  50 rows, TAPEX 25 rows, chatGPT 35 rows"""
    df = df.drop(['fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'age'],
                 axis='columns')
    df = df.rename({'marital.status': 'maritalStatus',
                    'hours.per.week': 'hoursPerWeek',
                    'native.country': 'nativeCountry',
                    }, axis='columns')
    df.columns = df.columns.str.lower()
    if sample_size is None:
        return df.sample(frac=1, random_state=random_state, ignore_index=True)

    return df.sample(sample_size, random_state=random_state, ignore_index=True)


def read_mushroom_dataset(df,
                          sample_size: int | None = None,
                          random_state: int = 2023) -> pd.DataFrame:
    """TAPAS  50 rows, Tapex 35 rows"""
    if sample_size is None:
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)
    else:
        df = df.sample(sample_size, random_state=random_state, ignore_index=True)
    cap_shape2name = {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat',
                      'k': 'knobbed', 's': 'sunken'}
    cap_surface2name = {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'}
    cap_color2wname = {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray',
                       'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white',
                       'y': 'yellow'}
    bruises2name = {'t': 'bruises', 'f': 'no'}
    odor2name = {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy',
                 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent',
                 's': 'spicy'}
    gill_attachment2name = {'a': 'attached', 'd': 'descending',
                            'f': 'free', 'n': 'notched'}
    gill_spacing2name = {'c': 'close', 'w': 'crowded', 'd': 'distant'}
    gill_size = {'b': 'broad', 'n': 'narrow'}
    gill_color = {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate',
                  'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink',
                  'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'}
    class2name = {'e': 'edible', 'p': 'poisonous'}

    # for each column in the df, map the dictionary
    df['cap-shape'] = df.apply(lambda row: cap_shape2name[row['cap-shape']], axis=1)
    df['cap-surface'] = df.apply(lambda row: cap_surface2name[row['cap-surface']], axis=1)
    df['cap-color'] = df.apply(lambda row: cap_color2wname[row['cap-color']], axis=1)
    df['bruises'] = df.apply(lambda row: bruises2name[row['bruises']], axis=1)
    df['odor'] = df.apply(lambda row: odor2name[row['odor']], axis=1)
    df['gill-attachment'] = df.apply(lambda row: gill_attachment2name[row['gill-attachment']], axis=1)
    df['gill-spacing'] = df.apply(lambda row: gill_spacing2name[row['gill-spacing']], axis=1)
    df['gill-size'] = df.apply(lambda row: gill_size[row['gill-size']], axis=1)
    df['gill-color'] = df.apply(lambda row: gill_color[row['gill-color']], axis=1)
    df['class'] = df.apply(lambda row: class2name[row['class']], axis=1)

    df = df.rename({'cap-shape': 'capShape',
                    'cap-surface': 'capSurface',
                    'cap-color': 'capColor',
                    'gill-attachment': 'gillAttachment',
                    'gill-spacing': 'gillSpacing',
                    'gill-size': 'gillSize',
                    'gill-color': 'gillColor',
                    }, axis='columns')
    df.columns = df.columns.str.lower()
    return df.loc[:, : 'gillcolor']


def check_model_names(model_name):
    model_name = model_name.lower()
    # semantic parsing models
    if model_name in ['resdsql', 'gap', 'skg'] or 'sp' in model_name:
        model_name = 'sp'
    return model_name


def read_data(db_id: str, model_name: str,
              input_base_path_data='./data',
              seed: int = 2023, inject_null_percentage: float = 0.0
              ) -> dict[str, pd.DataFrame]:
    model_name = check_model_names(model_name)

    sample_size = {
        ('medicine', 'chatgpt_qa', 'heart-attack'): 30,
        ('medicine', 'llama_qa', 'heart-attack'): 20,
        ('medicine', 'tapas', 'heart-attack'): 45,
        ('medicine', 'tapex', 'heart-attack'): 30,
        ('medicine', 'omnitab', 'heart-attack'): 20,
        ('medicine', 'sp', 'heart-attack'): None,
        ('medicine', 'tapas', 'breast-cancer'): 45,
        ('medicine', 'tapex', 'breast-cancer'): 30,
        ('medicine', 'chatgpt_qa', 'breast-cancer'): 35,
        ('medicine', 'llama_qa', 'breast-cancer'): 25,
        ('medicine', 'omnitab', 'breast-cancer'): 20,
        ('medicine', 'sp', 'breast-cancer'): None,

        ('ecommerce', 'tapas', 'sales-transactions'): 60,
        ('ecommerce', 'tapex', 'sales-transactions'): 20,
        ('ecommerce', 'chatgpt_qa', 'sales-transactions'): 40,
        ('ecommerce', 'llama_qa', 'sales-transactions'): 20,
        ('ecommerce', 'omnitab', 'sales-transactions'): 20,
        ('ecommerce', 'sp', 'sales-transactions'): 30000,
        ('ecommerce', 'tapas', 'fitness-trackers'): 50,
        ('ecommerce', 'tapex', 'fitness-trackers'): 20,
        ('ecommerce', 'chatgpt_qa', 'fitness-trackers'): 30,
        ('ecommerce', 'llama_qa', 'fitness-trackers'): 20,
        ('ecommerce', 'omnitab', 'fitness-trackers'): 20,
        ('ecommerce', 'sp', 'fitness-trackers'): None,

        ('finance', 'tapas', 'fraud'): 50,
        ('finance', 'tapex', 'fraud'): 25,
        ('finance', 'chatgpt_qa', 'fraud'): 30,
        ('finance', 'llama_qa', 'fraud'): 20,
        ('finance', 'omnitab', 'fraud'): 20,
        ('finance', 'sp', 'fraud'): 30000,
        ('finance', 'tapas', 'ibm'): 50,
        ('finance', 'tapex', 'ibm'): 20,
        ('finance', 'chatgpt_qa', 'ibm'): 25,
        ('finance', 'llama_qa', 'ibm'): 20,
        ('finance', 'omnitab', 'ibm'): 20,
        ('finance', 'sp', 'ibm'): None,

        ('miscellaneous', 'tapas', 'mush'): 50,
        ('miscellaneous', 'tapex', 'mush'): 25,
        ('miscellaneous', 'chatgpt_qa', 'mush'): 30,
        ('miscellaneous', 'llama_qa', 'mush'): 20,
        ('miscellaneous', 'omnitab', 'mush'): 20,
        ('miscellaneous', 'sp', 'mush'): None,
        ('miscellaneous', 'tapas', 'adult'): 50,
        ('miscellaneous', 'tapex', 'adult'): 20,
        ('miscellaneous', 'chatgpt_qa', 'adult'): 30,
        ('miscellaneous', 'llama_qa', 'adult'): 20,
        ('miscellaneous', 'omnitab', 'adult'): 20,
        ('miscellaneous', 'sp', 'adult'): 30000
    }

    if db_id == 'medicine':
        df_1 = read_heart_attack_dataset(
            pd.read_csv(f'{input_base_path_data}/medicine/heart-attack.csv'),
            sample_size=sample_size[(db_id, model_name, 'heart-attack')],
            random_state=seed
        )
        df_2 = read_breast_cancer_dataset(
            pd.read_csv(f'{input_base_path_data}/medicine/breast-cancer.csv'),
            sample_size=sample_size[(db_id, model_name, 'breast-cancer')],
            random_state=seed
        )
        db_tables = {'heartAttack': df_1, 'breastCancer': df_2}

    elif db_id == 'ecommerce':
        df_1 = read_sales_transactions_dataset(
            pd.read_csv(f'{input_base_path_data}/ecommerce/sales-transactions.csv'),
            sample_size=sample_size[(db_id, model_name, 'sales-transactions')],
            random_state=seed
        )

        df_2 = read_fitness_trackers_dataset(
            pd.read_csv(f'{input_base_path_data}/ecommerce/fitness-trackers.csv'),
            sample_size=sample_size[(db_id, model_name, 'fitness-trackers')],
            random_state=seed
        )
        db_tables = {'salesTransactions': df_1, 'fitnessTrackers': df_2}

    elif db_id == 'miscellaneous':
        df_1 = read_mushroom_dataset(
            pd.read_csv(f'{input_base_path_data}/miscellaneous/mushrooms.csv'),
            sample_size=sample_size[(db_id, model_name, 'mush')],
            random_state=seed
        )

        df_2 = read_adult_dataset(
            pd.read_csv(f'{input_base_path_data}/miscellaneous/adult-census.csv'),
            sample_size=sample_size[(db_id, model_name, 'adult')],
            random_state=seed
        )
        db_tables = {'mushrooms': df_1, 'adultCensus': df_2}

    elif db_id == 'finance':
        df_1 = read_bank_fraud_dataset(
            pd.read_csv(f'{input_base_path_data}/finance/account-fraud.csv'),
            sample_size=sample_size[(db_id, model_name, 'fraud')],
            random_state=seed
        )

        df_2 = read_finance_factory_ibm(
            pd.read_csv(f'{input_base_path_data}/finance/late-payment.csv'),
            sample_size=sample_size[(db_id, model_name, 'ibm')],
            random_state=seed
        )
        db_tables = {'accountFraud': df_1, 'latePayment': df_2}

    else:
        raise ValueError('Unknown dataset name')
    # inject null values
    db_tables = inject_null_values_in_tables(inject_null_percentage, db_tables, seed)
    return db_tables


def inject_null_values_in_tables(inject_null_percentage: float, tables: dict[str, pd.DataFrame], seed: int):
    """inject null into the tables"""
    np.random.seed(seed)
    if inject_null_percentage > 0.0:
        tables = {key: df.mask(np.random.random(df.shape) < inject_null_percentage)
                  for key, df in tables.items()}
    return tables


def combine_predictions_txt_with_df(prediction_path, df_path, prediction_col_name):
    df = pd.read_json(df_path)
    # read txt file
    predictions = pd.read_csv(prediction_path, delimiter='\\n', header=None)
    df[prediction_col_name] = predictions.values
    return df


def convert_to_list(x):
    if x is None:
        return None
    if x == '':
        return None
    return eval(x)


def convert_fk_index(data):
    fk_holder = []
    for fk in data["foreign_keys"]:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        ref_cid, cid = None, None
        tid = data['table_names_original'].index(tn)
        ref_tid = data['table_names_original'].index(ref_tn)

        for i, (tab_id, col_org) in enumerate(data['column_names_original']):
            if tab_id == ref_tid and ref_col == col_org:
                ref_cid = i
            elif tid == tab_id and col == col_org:
                cid = i
        if ref_cid and cid:
            fk_holder.append([cid, ref_cid])
    return fk_holder


def dump_db_json_schema_spider(db, f):
    '''read table and column info'''

    conn = sqlite3.connect(db)
    conn.execute('pragma foreign_keys=ON')
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {'db_id': f,
            'table_names_original': [],
            'table_names': [],
            'column_names_original': [(-1, '*')],
            'column_names': [(-1, '*')],
            'column_types': ['text'],
            'primary_keys': [],
            'foreign_keys': []}

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data['table_names_original'].append(table_name)
        data['table_names'].append(table_name.lower().replace("_", ' '))
        fks = conn.execute("PRAGMA foreign_key_list('{}') ".format(table_name)).fetchall()
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data['column_names_original'].append((i, col[1]))
            data['column_names'].append((i, col[1].lower().replace("_", " ")))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if 'char' in col_type or col_type == '' or 'text' in col_type or 'var' in col_type:
                data['column_types'].append('text')
            elif 'int' in col_type or 'numeric' in col_type or 'decimal' in col_type or 'number' in col_type \
                    or 'id' in col_type or 'real' in col_type or 'double' in col_type or 'float' in col_type:
                data['column_types'].append('number')
            elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                data['column_types'].append('time')
            elif 'boolean' in col_type:
                data['column_types'].append('boolean')
            else:
                data['column_types'].append('others')

            if col[5] == 1:
                data['primary_keys'].append(len(data['column_names']) - 1)

    data["foreign_keys"] = fk_holder
    data['foreign_keys'] = convert_fk_index(data)

    return data


def create_json_tables_file(db_path):
    db_files_sqlite = [(df + '.sqlite', df) for df in os.listdir(db_path) if
                       os.path.exists(os.path.join(db_path, df, df + '.sqlite'))]

    tables = []
    for f, df in db_files_sqlite:
        db = os.path.join(db_path, df, f)
        table = dump_db_json_schema_spider(db, df)
        table['column_names'] = table['column_names_original']
        table['table_names'] = table['table_names_original']
        tables.append(table)
    return tables
