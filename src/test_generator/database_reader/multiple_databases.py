import os

from .single_database import SingleDatabase


class MultipleDatabases:
    """
     TODO write documentation
    """

    def __init__(self, base_path_dbs: str,
                 _MAX_DB=15):
        """
        :param base_path_dbs:
        :param _MAX_DB:
        """
        self.base_path_dbs = base_path_dbs
        self.db_ids2db = dict()
        self._MAX_DB = _MAX_DB

    def __contains__(self, other: str):
        return any(other in db_id for db_id in self.db_ids2db)

    def __getitem__(self, key):
        if key not in self.db_ids2db:
            self.open_db(key)
        return self.db_ids2db[key]

    def open_db(self, db_id: str):
        if db_id not in self.db_ids2db:
            if len(self.db_ids2db) >= self._MAX_DB:
                # FIFO strategy if max reached
                self.db_ids2db.popitem()
            path = os.path.join(self.base_path_dbs,
                                'database',
                                db_id, f'{db_id}.sqlite')
            self.db_ids2db[db_id] = SingleDatabase(path, db_name=db_id)

    def get_table(self, key, tbl_name):
        """access the database, then return the requested table as pandas dataframe"""
        return self[key].get_table_from_name(tbl_name)

    def run_query(self, db_id: str, query: str) -> list:
        """run the query on the database and return the results in correct format"""
        ans_query = self[db_id].run_query(query)
        return [list(x) for x in ans_query]

    def close(self):
        """close the connection to all the databases"""
        [db.close() for db in self.db_ids2db.values()]
        self.db_ids2db = dict()
