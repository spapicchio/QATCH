import pandas as pd


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


def read_data(name: str, model_name: str, base_path='./data'):
    sample_size = {
        ('medicine', 'chatgpt', 'heart-attack'): 30,
        ('medicine', 'tapas', 'heart-attack'): 45,
        ('medicine', 'tapex', 'heart-attack'): 30,
        ('medicine', 'sp', 'heart-attack'): None,
        ('medicine', 'tapas', 'breast-cancer'): 45,
        ('medicine', 'tapex', 'breast-cancer'): 30,
        ('medicine', 'chatgpt', 'breast-cancer'): 35,
        ('medicine', 'sp', 'breast-cancer'): None,

        ('ecommerce', 'tapas', 'sales-transactions'): 60,
        ('ecommerce', 'tapex', 'sales-transactions'): 20,
        ('ecommerce', 'chatgpt', 'sales-transactions'): 40,
        ('ecommerce', 'sp', 'sales-transactions'): 30000,
        ('ecommerce', 'tapas', 'fitness-trackers'): 50,
        ('ecommerce', 'tapex', 'fitness-trackers'): 25,
        ('ecommerce', 'chatgpt', 'fitness-trackers'): 30,
        ('ecommerce', 'sp', 'fitness-trackers'): None,

        ('finance', 'tapas', 'fraud'): 50,
        ('finance', 'tapex', 'fraud'): 25,
        ('finance', 'chatgpt', 'fraud'): 30,
        ('finance', 'sp', 'fraud'): 30000,
        ('finance', 'tapas', 'ibm'): 50,
        ('finance', 'tapex', 'ibm'): 20,
        ('finance', 'chatgpt', 'ibm'): 25,
        ('finance', 'sp', 'ibm'): None,

        ('miscellaneous', 'tapas', 'mush'): 50,
        ('miscellaneous', 'tapex', 'mush'): 25,
        ('miscellaneous', 'chatgpt', 'mush'): 30,
        ('miscellaneous', 'sp', 'mush'): None,
        ('miscellaneous', 'tapas', 'adult'): 50,
        ('miscellaneous', 'tapex', 'adult'): 20,
        ('miscellaneous', 'chatgpt', 'adult'): 30,
        ('miscellaneous', 'sp', 'adult'): 30000
    }

    if name == 'medicine':
        df_1 = read_heart_attack_dataset(
            pd.read_csv(f'{base_path}/medicine/heart-attack.csv'),
            sample_size=sample_size[(name, model_name, 'heart-attack')])
        df_2 = read_breast_cancer_dataset(
            pd.read_csv(f'{base_path}/medicine/breast-cancer.csv'),
            sample_size=sample_size[(name, model_name, 'breast-cancer')])
        if model_name == 'sp':
            db_tables = {'heartAttack': df_1,
                         'breastCancer': df_2}
        else:
            db_tables = {'heart-attack': df_1,
                         'breast-cancer': df_2}

    elif name == 'ecommerce':
        df_1 = read_sales_transactions_dataset(
            pd.read_csv(f'{base_path}/ecommerce/sales-transactions.csv'),
            sample_size=sample_size[(name, model_name, 'sales-transactions')])

        df_2 = read_fitness_trackers_dataset(
            pd.read_csv(f'{base_path}/ecommerce/fitness-trackers.csv'),
            sample_size=sample_size[(name, model_name, 'fitness-trackers')])
        if model_name == 'sp':
            db_tables = {'salesTransactions': df_1,
                         'fitnessTrackers': df_2}
        else:
            db_tables = {'sales-transactions': df_1,
                         'fitness-trackers': df_2}

    elif name == 'miscellaneous':
        df_1 = read_mushroom_dataset(
            pd.read_csv(f'{base_path}/miscellaneous/mushrooms.csv'),
            sample_size=sample_size[(name, model_name, 'mush')])

        df_2 = read_adult_dataset(
            pd.read_csv(f'{base_path}/miscellaneous/adult-census.csv'),
            sample_size=sample_size[(name, model_name, 'adult')])
        if model_name == 'sp':
            db_tables = {'deadlyMushrooms': df_1,
                         'adultCensus': df_2}
        else:
            db_tables = {'deadly-mushrooms': df_1,
                         'adult-census': df_2}

    elif name == 'finance':
        df_1 = read_bank_fraud_dataset(
            pd.read_csv(f'{base_path}/finance/account-fraud.csv'),
            sample_size=sample_size[(name, model_name, 'fraud')])  # Tapex

        df_2 = read_finance_factory_ibm(
            pd.read_csv(f'{base_path}/finance/late-payment.csv'),
            sample_size=sample_size[(name, model_name, 'ibm')])  # Tapex
        if model_name == 'sp':
            db_tables = {'fraud': df_1,
                         'IBMLatePayment': df_2}
        else:
            db_tables = {'fraud': df_1,
                         'IBM-late-payment': df_2}
    else:
        raise ValueError('Unknown dataset name')
    return db_tables


def get_predictions(tbl_name, df, model_, tables, max_query_processing_num=1):
    table = tables[tbl_name]
    queries = df[df.tbl_name == tbl_name[1]].question.tolist()
    pred = model_.predict(table=table,
                          queries=queries,
                          max_query_processing_num=max_query_processing_num)
    return pred


def convert_to_list(x):
    if x is None:
        return None
    if x == '':
        return None
    return eval(x)
