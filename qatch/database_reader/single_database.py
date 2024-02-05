import logging
import os.path
import sqlite3

import pandas as pd


class SingleDatabase:
    """Provides a simplified interface for interacting with SQLite databases in Python.

    Attributes:
        conn (sqlite3.Connection): A connection object representing the SQLite database.
        cursor (sqlite3.Cursor): A cursor object used to execute SQL commands and retrieve results.
        db_name (str): The name of the database.
        table_schemas (Dict[str, pd.DataFrame]): Table name as key, table schema as value.
            Table schema is a Pandas DataFrame with columns: cid, name, type, notnull, dflt_value, pk.
        table_names (str): The names of the tables in the database.
        db_path (str): Path to the folder which contains the SQLite file.
        db_path_sqlite (str): Path to the SQLite file.
    """

    def __init__(self, db_path: str, db_name: str, tables: dict[str, pd.DataFrame] | None = None,
                 table2primary_key: dict[str, str] | None = None):
        if db_path == '' or db_path == '.' or db_path == './':
            db_path = os.getcwd()
        db_path = os.path.join(db_path, db_name)
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        path_sqlite_file = os.path.join(db_path, f'{db_name}.sqlite')
        # This creates a connection object that represents the database in case it does not exist
        conn = sqlite3.connect(path_sqlite_file)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        # A cursor is used to execute SQL commands and retrieve results.
        cursor = conn.cursor()
        # select all the table names from the database if any
        cursor.execute('SELECT name from sqlite_master where type= "table"')
        existing_tables = [tbl[0] for tbl in cursor.fetchall()]
        if len(existing_tables) == 0:
            if tables is None:
                # CASE 1: no tables provided and database does not exist
                # remove the created file
                conn.close()
                raise ValueError(f"Database path does not exist and no tables were provided "
                                 f"to create the new database in the given path:"
                                 f" {path_sqlite_file}")
            else:
                # CASE 2: tables provided and database does not exist
                logging.info(f"provided tables are stored in {path_sqlite_file}")
                existing_tables = list(tables.keys())
                self.set_tables_in_db(tables, conn, table2primary_key)
        else:
            if tables is not None:
                # CASE 3: database already exists and tables are provided
                # log warning tables provided but also database already exists
                logging.warning("tables provided but also database already exists, "
                                "tables in the database will be used")
        self.table2primary_key = table2primary_key
        self.db_path_sqlite = path_sqlite_file
        self.db_path = db_path
        self.db_name = db_name
        self.conn = conn
        self.cursor = cursor
        self.table_names = existing_tables
        self.table_schemas = {tbl_name: pd.read_sql_query(f'PRAGMA table_info("{tbl_name}")', self.conn)
                              for tbl_name in self.table_names}

    @staticmethod
    def set_tables_in_db(tables: dict[str, pd.DataFrame] | None,
                         conn: sqlite3.Connection,
                         table2primary_key: dict[str, str] | None):
        """
        Sets the tables in the SQLite database represented by the given connection object.

        This method takes a dictionary of tables in which keys are table names and values are Pandas DataFrames
        representing the tables, and sets these tables in the SQLite database represented by the `conn` connection object.

        The optional `table2primary_key` argument can be used to set primary keys for some or all tables. If not provided,
        all tables are created without primary keys.
        If the table contains an attribute with the same name of a primary key, a foreign key relationship is created.

        Note:
            - If a table is named as 'table', the method will replace its name with 'my_table'.
            - Assume the PKs have all different names. No two tables with the same PK name.

        Args:
            tables (Optional[Dict[str, pd.DataFrame]]): A dictionary of tables to set in the SQLite database.
                Keys are table names and values are corresponding Pandas DataFrames.

            conn (sqlite3.Connection): A connection object representing the SQLite database.

            table2primary_key (Optional[Dict[str, str]]): A dictionary mapping table names to primary keys.
                For example, if you want to set the primary key of table `A` to be `Key_1`, you should pass
                `table2primary_key={'A': 'Key_1'}`. Default is None.
        """
        for name, table in tables.items():
            if name == 'table':
                name = 'my_table'
            if not table2primary_key:
                table.to_sql(name, conn, if_exists='replace', index=False)
            else:
                create_table_string = SingleDatabase._create_table_in_db(name, table, table2primary_key)
                conn.cursor().execute(create_table_string)
                table.to_sql(name, conn, if_exists='append', index=False)

    @staticmethod
    def _create_table_in_db(name, table, table2primary_key):
        """
        Returns a SQLite CREATE TABLE command as a string, constructed based on the given table name, DataFrame, and primary_key2table dict.

        This method first converts pandas DataFrame dtypes to SQLite data types. Then, a SQLite CREATE TABLE command is built step by step.
        The command includes creating simple columns, adding primary keys, and creating foreign key relationships.
        Finally, the CREATE TABLE command is returned as a string.

        Args:
             name (str): The name of the table to be created in SQLite database.
             table (pd.DataFrame): A pandas DataFrame holding the data and structure of the SQL table. The dtype of each column translated to SQLite data types.
             table2primary_key (dict): A dictionary where the keys are the names of columns of the table, and the values are the names of the tables they are primary keys to.

        Example:
            >>> import pandas as pd
            >>> from your_module import SingleDatabase

            >>> name = 'sample_table'
            >>> table = pd.DataFrame({
            >>>    'id': [1, 2, 3],
            >>>    'name': ['Alice', 'Bob', 'Charlie'],
            >>>})
            >>>primary_key2table = {
            >>>    'sample_table_id': 'sample_table'
            >>>}

            >>>print(SingleDatabase._create_table_in_db(name,table,primary_key2table))
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
        [create_table.append(f'"{col}" {column2type[col]},') for col in table.columns]
        # Add primary key and foreign key
        for col in table.columns:
            if col in primary_key2table:
                if name == primary_key2table[col]:
                    # PRIMARY KEY ("Round"),
                    create_table.append(f'PRIMARY KEY ("{col}"),')
                else:
                    # FOREIGN KEY (`Winning_Aircraft`) REFERENCES `aircraft`(`Aircraft_ID`),
                    create_table.append(f'FOREIGN KEY (`{col}`) REFERENCES `{primary_key2table[col]}`(`{col}`),')
        # remove last comma
        create_table[-1] = create_table[-1][:-1]
        # add closing statement
        create_table.append(');')
        return " ".join(create_table)

    def get_table_given(self, table_name: str) -> pd.DataFrame:
        """
        Retrieves a specified table from the database as a Pandas DataFrame.

        Args:
            table_name (str): The name of the table to retrieve from the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the specified table from the database.
        """
        return pd.read_sql_query(f"SELECT * FROM {table_name}", self.conn)

    def get_schema_given(self, table_name: str) -> pd.DataFrame:
        """
        Given the table name, returns the schema of the table.
        Table schema is a Pandas DataFrame with columns: cid, name, type, notnull, dflt_value, pk.

        Args:
            table_name (str): The name of the table to retrieve the schema from the database.

        Returns:
            pd.DataFrame: A Pandas DataFrame representing the schema of the specified table from the database.
        """
        return self.table_schemas[table_name]

    def run_query(self, query: str) -> list[list]:
        """
        Run a query on the database and return the result.

        Args:
            query (str): The SQL query to be executed on the database.
        Returns:
            list[list]: A list of lists representing the result of the SQL query.
        """
        try:
            self.cursor.execute(query)
            output = self.cursor.fetchall()
            if not output:
                logging.warning(f'No query result for this query: {query}')
            return output
        except sqlite3.OperationalError as e:
            logging.error(f"Error while executing query: {query}")
            logging.error(e)
            raise
        # self.cursor.execute(query)
        # return self.cursor.fetchall()

    def close_connection(self):
        """Closes the connection to the database."""
        self.conn.close()
