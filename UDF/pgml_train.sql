create or replace function aisql.train_onnx(project_name text, task text, algorithm text, relation_name text,
                                            hyperparams text default '{}', search text default NULL,
                                            search_params text default '{}', search_args text default '{}',
                                            preprocess text default '{}')
                                            returns table(project_id integer, model_name text, algorithm text) as $$
import json
import numpy as np
import pickle
from cn_clip.global_def import (dataset_map, algorithm_map, cv_search_map, preprocessor_map,
                                convert_onnx_host, convert_onnx_port)
from cn_clip.to_onnx.ConvertONNXService import Client
from sklearn.model_selection import cross_validate
from thrift.protocol import TBinaryProtocol
from thrift.transport import TSocket
from thrift.transport import TTransport

plpy.info(project_name)
plpy.info(preprocess)
pp_map = json.loads(preprocess)

transport = TSocket.TSocket(host=convert_onnx_host, port=convert_onnx_port)
transport.setTimeout(30000)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)
client = Client(protocol)
try:
    transport.open()
    plpy.info(f"Connected to {convert_onnx_host}:{convert_onnx_port}")
except:
    plpy.info(f"Cannot connect to the ONNX conversion service at {convert_onnx_host}:{convert_onnx_port}")
    return [(-1, 'ONNX conversion service not found', algorithm)]

rv = plpy.execute("select nextval('aisql.onnx_prj_seq') as pid")
max_pid = rv[0]["pid"]
plan1 = plpy.prepare("insert into aisql.onnx_projects(id, project_name, relation_name, model_names, score) " +
                     "values($1, $2, $3, $4, $5)", ["integer", "varchar", "varchar", "varchar[]", "float8"])
plan2 = plpy.prepare("insert into aisql.onnx_prj_models(project_id, model_name, algorithm, model_data) " +
                     "values($1, $2, $3, $4)", ["integer", "varchar", "varchar", "bytea"])

# plpy.info(model_name)
# plpy.info(hyperparams)
# plpy.info(search_params)
# plpy.info(search_args)
if relation_name in dataset_map:
    dataset = dataset_map[relation_name]()
    X, y = dataset.data, dataset.target
elif task == "clustering":
    rv = plpy.execute(f"select data from {relation_name}")
    data = []
    for r in rv:
        data.append(np.array(r["data"]))
    X, y = np.array(data), None
else:
    rv = plpy.execute(f"select data, target from {relation_name}")
    data, target = [], []
    for r in rv:
        data.append(np.array(r["data"]))
        target.append(r["target"])
    X, y = np.array(data), np.array(target)

model_names = [f"{task}:{algorithm}"]
model_recs = []

for pname, params in pp_map.items():
    preprocessor = preprocessor_map[pname]
    if pname == "min_max" and "feature_range" in params:
        params["feature_range"] = tuple(params["feature_range"])
    if pname == "robust" and "quantile_range" in params:
        params["quantile_range"] = tuple(params["quantile_range"])
    pp = preprocessor(**params)
    pp.fit(X, y)
    alg_name = pname + ":" + json.dumps(params)
    # onnx = convert_sklearn(pp, pname, [("features", FloatTensorType([None, X.shape[1]]))])
    # bin_data = onnx.SerializeToString()
    bin_data = client.convert_sklearn(pickle.dumps(pp), pname, X.shape[1])
    plpy.execute(plan2, [max_pid, pname, alg_name, bin_data])
    model_names.append(pname)
    model_recs.append((max_pid, pname, alg_name))
    X = pp.transform(X)

alg = algorithm_map[task][algorithm](**json.loads(hyperparams))
if algorithm == "catboost":
    test_score = 10  # Todo for CatBoost, use CatBoost as default
elif search is None:
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

# plpy.info(algorithm)
if algorithm == "catboost":
    alg.fit(X, y, cat_features=[], silent=True)
else:
    alg.fit(X, y)
model = pickle.dumps(alg)
if algorithm == "xgboost":
    # onnx = convert_xgboost(alg, algorithm, [("features", FloatTensorType([None, X.shape[1]]))])
    bin_data = client.convert_xgboost(model, algorithm, X.shape[1])
elif algorithm == "lightgbm":
    # onnx = convert_lightgbm(alg, algorithm, [("features", FloatTensorType([None, X.shape[1]]))])
    bin_data = client.convert_lightgbm(model, algorithm, X.shape[1])
elif algorithm == "catboost":
    # onnx = convert_catboost(alg, algorithm)
    bin_data = client.convert_catboost(model, algorithm)
else:
    # onnx = convert_sklearn(alg, algorithm, [("features", FloatTensorType([None, X.shape[1]]))])
    bin_data = client.convert_sklearn(model, algorithm, X.shape[1])
# bin_data = onnx.SerializeToString()
plpy.execute(plan1, [max_pid, project_name, relation_name, model_names, test_score])

model_name = f"{task}:{algorithm}"
alg_name = f"{algorithm}:{hyperparams}"
plpy.execute(plan2, [max_pid, model_name, alg_name, bin_data])
model_recs.append((max_pid, model_name, alg_name))

return model_recs
$$ language 'plpython3u';
