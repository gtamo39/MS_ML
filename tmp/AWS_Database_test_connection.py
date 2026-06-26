import psycopg2

db_host = 'seracbio-dev.cfyi0icu0fkt.eu-north-1.rds.amazonaws.com'
db_name = 'postgres'
db_user = 'seracbio'
db_pass = 'Bunker.Regal8'

connection = psycopg2.connect(host= db_host, database= db_name,
                              user= db_user, password= db_pass)
#check if you are connected with the database
print("Connected to the database")

cursor = connection.cursor()
cursor.execute('SELECT version()')
db_version = cursor.fetchone()
#check the version of the postgreSQL
print(db_version)

cursor.execute("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_type = 'BASE TABLE'
      AND table_schema NOT IN ('pg_catalog', 'information_schema')
    ORDER BY table_schema, table_name;
""")

tables = cursor.fetchall()

print("Tables:")
for schema, table in tables:
    print(f"{schema}.{table}")

table_name = "maintable"

cursor.execute(f"SELECT * FROM {table_name} ")

rows = cursor.fetchall()
#print the table selected
print(f"\n rows from {table_name}:")
for row in rows:
    print(row)

cursor.close()

