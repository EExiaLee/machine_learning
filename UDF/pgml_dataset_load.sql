create or replace function make_json_map(kv_array text[][]) returns text as $$
result = []
for kv in kv_array:
    result.append(f"{kv[0]}:{kv[1]}")
return '{' + ','.join(result) + '}'
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

create or replace function aisql.load_dataset(ds_name varchar, name varchar default '')
    returns table(tb_name text, size integer) as $$
import numpy
from cn_clip.global_def import dataset_map

if len(name) == 0:
    tb_name = f"aisql.{ds_name}"
else:
    tb_name = name

if ds_name not in dataset_map:
    return [('ERROR', -1)]

plpy.execute(f"drop table if exists {tb_name}")
plpy.execute(f"create table {tb_name}(id serial primary key, data float8[], label integer)")
plan = plpy.prepare(f"insert into {tb_name}(data, label) values($1, $2)", ["float8[]", "integer"])
ds = dataset_map[ds_name]()
target = ds.target.astype(numpy.int32)
lng = len(target)
for i in range(lng):
    plpy.execute(plan, [ds.data[i], target[i]])
return [(tb_name, lng)]
$$ language 'plpython3u';
