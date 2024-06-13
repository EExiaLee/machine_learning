create or replace function aisql.predict_onnx(project_name text, tuple float8[]) returns float8 as $$
import json
import numpy as np
import onnxruntime as rt
from cn_clip.global_def import global_dict

plan = plpy.prepare("select id,model_names from aisql.onnx_projects where project_name=$1 order by score desc limit 1",
                    ["varchar"])
rv = plpy.execute(plan, [project_name])
if len(rv) == 0:
    key = f"onnx:{project_name}:not_found"
    if key not in global_dict:
        global_dict[key] = True
        plpy.info("Not found " + project_name)
    return None

pid = rv[0]["id"]
model_names = rv[0]["model_names"]
plan = plpy.prepare("select model_data from aisql.onnx_prj_models where project_id=$1 and model_name=$2",
                    ["integer", "varchar"])
providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in rt.get_available_providers() else []
providers.append('CPUExecutionProvider')

data = np.asarray([tuple], dtype=np.float32)
for model_name in model_names[1:]:
    key = f"onnx:{project_name}:{model_name}"
    if key in global_dict:
        sess = global_dict[key]
    else:
        rv = plpy.execute(plan, [pid, model_name])
        model = rv[0]["model_data"]
        sess = rt.InferenceSession(model, providers=providers)
        global_dict[key] = sess
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    data = sess.run([output_name], {input_name: data})[0]

model_name = model_names[0]
key = f"onnx:{project_name}:{model_name}"
if key in global_dict:
    sess = global_dict[key]
else:
    rv = plpy.execute(plan, [pid, model_name])
    model = rv[0]["model_data"]
    sess = rt.InferenceSession(model, providers=providers)
    global_dict[key] = sess

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred = sess.run([label_name], {input_name: data})[0]
return pred.flatten()[0]
$$ language 'plpython3u';
