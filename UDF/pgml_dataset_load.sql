create or replace function make_json_map(kv_array text[][]) returns text as $$
result = []
for kv in kv_array:
    result.append(f"{kv[0]}:{kv[1]}")
return '{' + ','.join(result) + '}'
$$ language plpython3u;

create or replace function make_kv_as_map(k text, v text, quote_val boolean default FALSE) returns text as $$
key = '"' + k.replace('"', '\\"') + '"'
val = '"' + v.replace('"', '\\"') + '"' if quote_val else v
return '{' + f"{key}:{val}" + '}'
$$ language plpython3u;

create or replace function make_kv(k text, v text, quote_val boolean default FALSE) returns text[] as $$
key = '"' + k.replace('"', '\\"') + '"'
val = '"' + v.replace('"', '\\"') + '"' if quote_val else v
return [key, val]
$$ language plpython3u;

create or replace function make_kv(k text, vs text[], quote_val boolean default FALSE) returns text[] as $$
key = '"' + k.replace('"', '\\"') + '"'
values = []
for v in vs:
    values.append('"' + v.replace('"', '\\"') + '"' if quote_val else v)
val = '[' + ','.join(values) + ']'
return [key, val]
$$ language plpython3u;

create or replace function make_kv_array(ks text[], vs text[], quote_val boolean default FALSE) returns text[][] as $$
pairs = []
for k, v in zip(ks, vs):
    key = '"' + k.replace('"', '\\"') + '"'
    val = '"' + v.replace('"', '\\"') + '"' if quote_val else v
    pairs.append([key, val])
return pairs
$$ language plpython3u;

create or replace function make_kv_ext_array(ks text[], vs text[][], quote_val boolean default FALSE) returns text[][] as $$
pairs = []
for k, vv in zip(ks, vs):
    key = '"' + k.replace('"', '\\"') + '"'
    values = []
    for v in vv:
        values.append('"' + v.replace('"', '\\"') + '"' if quote_val else v)
    val = '[' + ','.join(values) + ']'
    pairs.append([key, val])
return pairs
$$ language plpython3u;

create or replace function aisql.initialize_ml() returns void as $$
from cn_clip.ml_utils import initialize_ml
initialize_ml(plpy)
$$ language 'plpython3u';

create or replace function aisql.load_dataset(ds_name varchar, name varchar default '')
    returns table(tb_name text, size integer) as $$
import numpy
from cn_clip.ml_utils import generate_dataset
return generate_dataset(plpy, ds_name, name)
$$ language 'plpython3u';

create or replace function aisql.load_imdb_dataset(data_dir varchar, part varchar)
    returns table(tb_name text, size integer) as $$
import numpy
from cn_clip.ml_utils import generate_imdb_data
tb_name = f"aisql.imdb_{part}"
plpy.execute(f"drop table if exists {tb_name}")
plpy.execute(f"create table {tb_name}(comment text, label integer)")
count = generate_imdb_data(plpy, data_dir, part)
return [(tb_name, count)]
$$ language 'plpython3u';
