import pandas as pd


def utils_convert_df_in_sql_code(name: str, table: pd.DataFrame, table2primary_key: dict):
    """
    Returns a SQLite CREATE TABLE command as a string, constructed based on the given table name, DataFrame,
    and primary_key2table dict.

    This method first converts pandas DataFrame dtypes to SQLite data types.
    Then, a SQLite CREATE TABLE command is built step by step.
    The command includes creating simple columns, adding primary keys, and creating foreign key relationships.
    Finally, the CREATE TABLE command is returned as a string.

    Args:
         name (str): The name of the table to be created in SQLite database.
         table (pd.DataFrame): A pandas DataFrame holding the data and structure of the SQL table.
                              The dtype of each column translated to SQLite data types.
         table2primary_key (dict): A dictionary where the keys are the names of columns of the table,
                                    and the values are the names of the tables they are primary keys to.

    Example:
        >>> import pandas as pd
        >>> from qatch.database_reader import SingleDatabase

        >>> name = 'sample_table'
        >>> table = pd.DataFrame({
        >>>    'id': [1, 2, 3],
        >>>    'name': ['Alice', 'Bob', 'Charlie'],
        >>>})
        >>>primary_key2table = {
        >>>    'id': 'sample_table'
        >>>}
        >>>print(utils_convert_df_in_sql_code(name, table, primary_key2table))
        >>># Outputs: CREATE TABLE `sample_table`( "id" INTEGER, "name" TEXT, PRIMARY KEY ("id") );
    """

    def convert_pandas_dtype_to_sqlite_type(type_):
        if 'int' in type_:
            return 'INTEGER'
        if 'float' in type_:
            return 'REAL'
        if 'object' in type_ or 'date' in type_:
            return 'TEXT'

    primary_key2table = {tbl_PK: tbl_name for tbl_name, tbl_PK in
                         table2primary_key.items()} if table2primary_key else None
    column2type = {k: convert_pandas_dtype_to_sqlite_type(str(table.dtypes[k]))
                   for k in table.dtypes.index}
    create_table = [f'CREATE TABLE `{name}`(']
    # add simple col
    # "Round" real,
    [create_table.append(f'`{col}` {column2type[col]},') for col in table.columns]
    # Add primary key and foreign key
    for col in table.columns:
        if col in primary_key2table:
            if name == primary_key2table[col]:
                # PRIMARY KEY ("Round"),
                create_table.append(f'PRIMARY KEY (`{col}`),')
            else:
                # FOREIGN KEY (`Winning_Aircraft`) REFERENCES `aircraft`(`Aircraft_ID`),
                create_table.append(f'FOREIGN KEY (`{col}`) REFERENCES `{primary_key2table[col]}`(`{col}`),')
    # remove last comma
    create_table[-1] = create_table[-1][:-1]
    # add closing statement
    create_table.append(');')
    return " ".join(create_table)
