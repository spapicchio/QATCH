import logging

import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm


def inject_ambiguity(tests_df, ambiguity_annotation_file='data/workshop/ambiguity_annotation_task_1.xlsx',
                     ambiguity_th=2.0):
    """This function is the entry point that will perform the ambiguity injection."""
    name2task_1 = read_annotation_file(ambiguity_annotation_file)

    pbar = tqdm(name2task_1.items(), desc='Analyzing')
    dbs = []

    for db_name, ambiguity_tbl in pbar:
        pbar.set_description(f'Analyzing {db_name}')
        db_tests = process_tbl(db_name, ambiguity_tbl, ambiguity_th, tests_df)
        dbs.append(db_tests)

    return pd.concat(dbs).reset_index(drop=True)


def process_tbl(db_name, ambiguity_tbl, ambiguity_th, tests_df):
    """This function performs the main operation of reading and saving the tests for each database."""

    # Adjusting the database name
    db_name = adjust_db_name(db_name)

    # Getting the ambiguous columns
    ambiguous_col = get_ambiguous_col(ambiguity_tbl)

    # Getting the test data for this database
    db_tests = get_db_tests(db_name, tests_df)

    # Filtering out the ambiguous test cases
    db_tests = filter_ambiguous_tests(ambiguous_col, ambiguity_tbl, ambiguity_th, db_tests)

    # get the ambiguous labels associated with the associated cols
    label2attribute = extract_labels_above_threshold(ambiguity_tbl, ambiguity_th)

    # If there are any ambiguous test cases
    if not db_tests.empty:
        # Extracting all the necessary fields and aggregating similar questions
        new_cols = ['ambiguous_questions', 'swapped_attribute', 'ambiguous_labels', 'ambiguous_attributes']
        db_tests[new_cols] = db_tests.apply(
            lambda x: create_ambiguous_cols(x.question, ambiguity_tbl,
                                            label2attribute, ambiguity_th),
            axis=1
        )

        db_tests = db_tests.explode(new_cols)
        db_tests = aggregate_similar_questions(db_tests)
    else:
        logging.warning(f'No ambiguous tests for {db_name}')

    return db_tests


def create_ambiguous_cols(question: str, ambiguity_tbl: pd.DataFrame, label2attribute: dict[str, list],
                          ambiguity_th: str) -> pd.Series:
    """
    Identifies ambiguous questions and maps them to their respective labels and attributes.

    Args:
        question (str): The question in consideration
        ambiguity_tbl (pd.DataFrame): DataFrame representing the ambiguity table
        label2attribute  (cict[str, list]): Dictionary where keys are labels, values are lists of attributes
        where corresponding values in ambiguity table are above threshold
        ambiguity_th: Ambiguity threshold

    Returns:
        pd.Series: Pandas Series containing lists of questions, swapped_attributes, labels, and ambiguous_attributes
    """
    # Initialize lists which will store the final results.
    ambiguous_attributes = [None]
    labels = [None]
    swapped_attributes = [None]
    questions = [question]

    # Go through each column in the ambiguity table.
    for col in ambiguity_tbl:
        if col == 'label':
            continue
        # If the column name appears in the question, then we need to process the ambiguity.
        if col in question:
            mask = ambiguity_tbl.loc[:, col] >= ambiguity_th  # Identify rows where current column value >= 2.0
            ambiguous_label_index = ambiguity_tbl.loc[mask, col].index
            ambiguous_labels = ambiguity_tbl.loc[ambiguous_label_index, 'label'].to_list()

            # If there are any ambiguous labels, then we update all the result lists correspondingly.
            if ambiguous_labels:
                # Extend the list of attributes with the attributes corresponding to found ambiguous labels.
                ambiguous_attributes.extend([label2attribute[x] for x in ambiguous_labels])
                labels.extend(ambiguous_labels)  # Extend the list of labels with found ambiguous labels.
                # Generate new questions by replacing the current attribute name (column name) with each found
                # ambiguous label and extend the list of questions.
                questions.extend([question.replace(col, x) for x in ambiguous_labels])
                swapped_attributes.extend([col] * len(ambiguous_labels))  # Extend the list of swapped attributes.

    # Check that lists have the same lengths. This asserts the consistency of the result.
    assert len(questions) == len(labels) == len(ambiguous_attributes) == len(swapped_attributes)
    output = pd.Series({
        'question': questions,
        'swapped_attribute': swapped_attributes,
        'ambiguous_labels': labels,
        'ambiguous_attributes': ambiguous_attributes
    })
    return output


def read_annotation_file(file_name):
    """read the ambiguity annotation excell file and returns a dictionary
    where the key is the filenae and the value is the pandas dataframe"""
    sheet_name2dataframe = pd.read_excel(file_name, sheet_name=None)
    return {name.replace('-', '_'): sheet_name2dataframe[name].iloc[:, 1:] for name in sheet_name2dataframe}



def adjust_db_name(db_name):
    """adjust the database names"""
    if db_name == 'acronyms':
        return 'basket_acronyms'
    elif db_name == 'full':
        return 'basket_full'
    else:
        return db_name


def get_ambiguous_col(ambiguity_tbl):
    """return the ambiguous columns from the ambiguity table."""
    return list(ambiguity_tbl.columns[2:])


def get_db_tests(db_name, tests_df):
    """get the tests for a particular database from the main tests dataframe."""
    return tests_df[tests_df.db_id == db_name]


def filter_ambiguous_tests(ambiguous_col, ambiguity_tbl, ambiguity_th, db_tests):
    """filter out the tests that do not contain ambiguous queries."""
    mask = db_tests['query'].apply(
        lambda x: any([col in x for col in ambiguous_col if any(ambiguity_tbl.loc[:, col] >= ambiguity_th)]))
    return db_tests[mask]


def extract_labels_above_threshold(ambiguity_tbl: pd.DataFrame, ambiguity_th: float) -> dict[str, list]:
    """
    Extracts labels from the ambiguity dataframe that are above the provided threshold

    Args:
        ambiguity_tbl (pd.DataFrame): DataFrame representing the ambiguity table
        ambiguity_th (float): Threshold value for ambiguity

    Returns:
        Dict[str, list]: Dictionary where keys are labels, values are lists of attributes
                         where corresponding values in ambiguity table are above threshold
    """
    # Build a dictionary where the keys are labels and the values are a list of attributes
    # where the corresponding values in the table are greater or equal to AMBIGUITY_SUM.
    label2attribute = ambiguity_tbl.set_index('label').to_dict('index')
    label2attribute = {label: [attr for attr, val in attributes.items() if val >= ambiguity_th]
                       for label, attributes in label2attribute.items()}
    return label2attribute


def aggregate_similar_questions(df):
    """
       Main function to aggregate similar questions.
       Parameters:
         -- df: Dataframe
    """
    df['target_queries'] = df.apply(create_target_queries, axis=1)
    df = drop_unnecessary_rows(df)
    return df


def create_target_queries(row):
    """
       Function to create target queries.
       Parameters:
         -- row: a dataframe row
       This function will return target queryies by replacing swapped_attribute with attributes from the row.
    """
    if row['swapped_attribute']:
        return [row['query'].replace(row['swapped_attribute'], attr) for attr in row['ambiguous_attributes']]
    return None


def drop_unnecessary_rows(df):
    """
       Function to drop unnecessary rows.
       Parameters:
         -- df: Dataframe
       This function returns a dataframe after dropping duplicate ambiguous questions and rows
       with 'swapped_attribute' as None
    """
    df = df.drop_duplicates('ambiguous_questions')
    df = df.dropna(subset=['swapped_attribute'])
    return df
