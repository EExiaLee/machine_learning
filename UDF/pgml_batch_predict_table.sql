create or replace function aisql.batch_predict(project_name text, tuple_text text) returns text[] as $$
import json
import numpy as np
import pickle
from cn_clip.global_def import global_dict

tuple = json.loads(tuple_text)
dimension = None
for k, vs in tuple.items():
    tuple[k] = np.array(vs).reshape(-1, 1)
    if dimension is None:
        dimension = len(vs)

plan = plpy.prepare("select id,x_column_names,y_column_name,model_names from aisql.projects where project_name=$1 order by score desc limit 1",
                    ["varchar"])
rv = plpy.execute(plan, [project_name])
if len(rv) == 0:
    key = f"{project_name}:not_found"
    if key not in global_dict:
        global_dict[key] = True
        plpy.info("Not found " + project_name)
    return None

pid = rv[0]["id"]
model_names = rv[0]["model_names"]
x_names = rv[0]["x_column_names"]
y_name = rv[0]["y_column_name"]
plan = plpy.prepare("select model_data from aisql.prj_models where project_id=$1 and model_name=$2",
                    ["integer", "varchar"])

y_encoder = None
for model_name in model_names[1:]:
    key = f"{project_name}:{model_name}"
    if y_name is not None and model_name == f"{y_name}:encode":
        if key in global_dict:
            y_encoder = global_dict[key]
        else:
            rv = plpy.execute(plan, [pid, model_name])
            y_encoder = pickle.loads(rv[0]["model_data"])
            global_dict[key] = y_encoder
    else:
        if key in global_dict:
            model = global_dict[key]
        else:
            rv = plpy.execute(plan, [pid, model_name])
            model = pickle.loads(rv[0]["model_data"])
            global_dict[key] = model
        x_name, pp_name = model_name.split(':')
        if x_name in tuple:
            tuple[x_name] = model.transform(tuple[x_name])
        elif pp_name == "impute":
            tuple[x_name] = model.transform(np.array([[np.nan]]))

columns = []
for x_name in x_names:
    if x_name in tuple:
        columns.append(tuple[x_name])
    else:
        columns.append([[np.nan]] * dimension)
# plpy.info(columns)
X = np.concatenate(columns, axis=1)

model_name = model_names[0]
key = f"{project_name}:{model_name}"
if key in global_dict:
    model = global_dict[key]
else:
    rv = plpy.execute(plan, [pid, model_name])
    model = pickle.loads(rv[0]["model_data"])
    global_dict[key] = model

label = model.predict(X) if y_encoder is None else y_encoder.inverse_transform(model.predict(X))
return [str(v) for v in label.flatten()]
$$ language 'plpython3u';
