create or replace function aisql.train(project_name text, task text, algorithm text, relation_name text,
                                       x_column_names text[], y_column_name text default NULL,
                                       cat_columns integer[] default NULL, hyperparams text default '{}',
                                       search text default NULL, search_params text default '{}',
                                       search_args text default '{}', preprocess text default '{}')
                                       returns table(project_id integer, model_name text, algorithm text) as $$
import json
import numpy as np
import pickle
from cn_clip.global_def import algorithm_map, cv_search_map, preprocessor_map
from sklearn.model_selection import cross_validate
from skl2onnx.common.data_types import FloatTensorType
from sklearn.preprocessing import LabelEncoder

uniform_x = "_ALL_X_"
plpy.info(project_name)
pp_map = json.loads(preprocess)
all_map = pp_map.get(uniform_x)

rv = plpy.execute("select nextval('aisql.project_seq') as pid")
max_pid = rv[0]["pid"]
plan1 = plpy.prepare("insert into aisql.projects(id, project_name, relation_name, x_column_names, y_column_name, " +
                     "cat_columns, model_names, score) values($1, $2, $3, $4, $5, $6, $7, $8)",
                     ["integer", "varchar", "varchar", "varchar[]", "varchar", "smallint[]", "varchar[]", "float8"])
plan2 = plpy.prepare("insert into aisql.prj_models(project_id, model_name, algorithm, model_data) values($1, $2, $3, $4)",
                     ["integer", "varchar", "varchar", "bytea"])
model_names = [f"{task}:{algorithm}"]
model_recs = []

if y_column_name is None or task == 'clustering':
    y = None
else:
    rv = plpy.execute(f"select array_agg({y_column_name}) from {relation_name}")
    y = np.array(rv[0]['array_agg'])
    if y_column_name in pp_map:
        y_map = pp_map[y_column_name]
        if "encode" in y_map:
            encode_obj = y_map["encode"]
            encoder = LabelEncoder()
            categories = encode_obj["categories"] if isinstance(encode_obj, dict) and "categories" in encode_obj else None
            if categories is None:
                encoder.fit(y)
            else:
                encoder.fit(categories)
            model_name = f"{y_column_name}:encode"
            alg_name = "encode:" + json.dumps(encode_obj)
            bin_data = pickle.dumps(encoder)
            plpy.execute(plan2, [max_pid, model_name, alg_name, bin_data])
            model_names.append(model_name)
            model_recs.append((max_pid, model_name, alg_name))
            y = encoder.transform(y)

columns = []
for index in range(len(x_column_names)):
    x_name = x_column_names[index]
    rv = plpy.execute("select array_agg({}) from {}".format(x_name, relation_name))
    x = np.array(rv[0]['array_agg']).reshape(-1, 1)
    if x_name in pp_map:
        x_map = pp_map[x_name]
    elif all_map is not None:
        x_map = all_map
    else:
        x_map = None
    if x_map is not None:
        for pname, params in x_map.items():
            preprocessor = preprocessor_map[pname]
            if pname == "min_max" and "feature_range" in params:
                params["feature_range"] = tuple(params["feature_range"])
            if pname == "robust" and "quantile_range" in params:
                params["quantile_range"] = tuple(params["quantile_range"])
            if pname == "one_hot":
                params["sparse_output"] = False
            pp = preprocessor(**params)
            pp.fit(x, y)
            model_name = f"{x_name}:{pname}"
            alg_name = pname + ":" + json.dumps(params)
            bin_data = pickle.dumps(pp)
            plpy.execute(plan2, [max_pid, model_name, alg_name, bin_data])
            model_names.append(model_name)
            model_recs.append((max_pid, model_name, alg_name))
            x = pp.transform(x)
    columns.append(x)
X = np.concatenate(columns, axis=1)

alg = algorithm_map[task][algorithm](**json.loads(hyperparams))
if cat_columns is None:
    if search is None:
        scores = cross_validate(alg, X, y)
        test_score = np.average(scores['test_score'])
    else:
        args = json.loads(search_args)
        if search == "random" and "n_iter" in args:
            cv_search = cv_search_map[search](alg, json.loads(search_params), n_iter=args["n_iter"])
        else:
            cv_search = cv_search_map[search](alg, json.loads(search_params))
        cv_search.fit(X, y)
        test_score = cv_search.best_score_
        best_param = cv_search.best_params_
        alg.set_params(**best_param)
else:
    test_score = 10  # Todo for CatBoost, use CatBoost as default

if cat_columns is None:
    plpy.execute(plan1, [max_pid, project_name, relation_name, x_column_names, y_column_name, [], model_names, test_score])
    # pickle.dump(X, open('D:/du-work/autotvm_tutorial/nursery_X.pkl', 'wb'))
    # pickle.dump(y, open('D:/du-work/autotvm_tutorial/nursery_y.pkl', 'wb'))
    alg.fit(X, y)
else:
    plpy.execute(plan1, [max_pid, project_name, relation_name, x_column_names, y_column_name, cat_columns, model_names, test_score])
    alg.fit(X, y, cat_features=cat_columns, silent=True)

bin_data = pickle.dumps(alg)
model_name = f"{task}:{algorithm}"
alg_name = f"{algorithm}:{hyperparams}"
plpy.execute(plan2, [max_pid, model_name, alg_name, bin_data])
model_recs.append((max_pid, model_name, alg_name))

return model_recs
$$ language 'plpython3u';
