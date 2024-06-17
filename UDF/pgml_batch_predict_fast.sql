create or replace function aisql.batch_predict_fast(project_name text, tuples float8[][]) returns float8[] as $$
import json
import numpy as np
import pickle
from cn_clip.global_def import global_dict

plan = plpy.prepare("select id,model_names from aisql.fast_projects where project_name=$1 order by score desc limit 1",
                    ["varchar"])
rv = plpy.execute(plan, [project_name])
if len(rv) == 0:
    key = f"fast:{project_name}:not_found"
    if key not in global_dict:
        global_dict[key] = True
        plpy.info("Not found " + project_name)
    return None

pid = rv[0]["id"]
model_names = rv[0]["model_names"]
plan = plpy.prepare("select model_data from aisql.fast_prj_models where project_id=$1 and model_name=$2",
                    ["integer", "varchar"])

data = np.asarray(tuples, dtype=np.float32)
for model_name in model_names[1:]:
    key = f"fast:{project_name}:{model_name}"
    if key in global_dict:
        model = global_dict[key]
    else:
        rv = plpy.execute(plan, [pid, model_name])
        bin_data = rv[0]["model_data"]
        model = pickle.loads(bin_data)
        global_dict[key] = model
    data = model.transform(data)
    # plpy.info(data.shape)

model_name = model_names[0]
key = f"fast:{project_name}:{model_name}"
if key in global_dict:
    model = global_dict[key]
else:
    rv = plpy.execute(plan, [pid, model_name])
    bin_data = rv[0]["model_data"]
    model = pickle.loads(bin_data)
    global_dict[key] = model

pred = model.predict(data.astype(np.float16))
return pred.flatten()
$$ language 'plpython3u';
