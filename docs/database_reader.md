## Overview

The ***database_reader*** package handles connections with the databases file.

It contains two class:

- **SingleDatabase**: Given a dictionary of pandas dataframe, it creates the database folder with the database sqlite file.
- **MultipleDatabases**: Given the path where the databases are stored, it automatically creates the connection with the stored file. 

If the data is already stored in database sqlite files following this pattern "db_save_path/db_id/db_id.sqlite"
you can create a connection between the tool and the files with *MultipleDatabase* class:

```python
from qatch.database_reader import MultipleDatabases

# The path to multiple databases
db_save_path = 'test_db'
databases = MultipleDatabases(db_save_path)
```

Instead, if you want to specify different data you can use the *SingleDatabase* class 
to create the sqlite databases in "db_save_path/db_id/db_id.sqlite"

Assume the PKs have all different names. No two tables with the same PK name.

```python
import pandas as pd

from qatch.database_reader import SingleDatabase

# Create dummy table
data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# define the tables in the database (<table_name> : <table>)
db_tables = {'olympic_games': table}

# define where to store the sqlite database
db_save_path = 'test_db'

# define the name of the database
db_id = 'olympic'

# define the PK
# Assume the PKs have all different names. Two tables cannot have same PK name.
table2primary_key = {'olympic_games': 'id'}

# create database connection
db = SingleDatabase(db_path=db_save_path, db_name=db_id, tables=db_tables, table2primary_key=table2primary_key)
```


::: qatch.database_reader