CREATE EXTENSION plpython3u;

create schema aisql;

set search_path to aisql;

-- select make_json_map(array[array[quote_str('a'),quote_str('1')], array[quote_str('b'),'2']]);



select make_json_map(array[make_kv('a', '1', TRUE), make_kv('b', '2')]);



select make_json_map(array[make_kv('a', '1'), make_kv('b', string_to_array('1,2,3,4', ','), TRUE)]);



select make_json_map(make_kv_array(string_to_array('a,b,c,d', ','), string_to_array('1,2,3,4', ',')));



select make_json_map(make_kv_ext_array(string_to_array('a,b,c,d', ','), string_to_array('1,2,3,4', ',')));



create sequence aisql.project_seq;



create table aisql.projects(id int primary key, project_name varchar, relation_name varchar, x_column_names varchar[], y_column_name varchar, cat_columns smallint[], model_names varchar[], score float8);



create index project_name_index on aisql.projects(project_name);



create table aisql.prj_models(project_id int, model_name varchar, algorithm varchar, model_data bytea, primary key (project_id, model_name));



create sequence aisql.onnx_prj_seq;



create table aisql.onnx_projects(id int primary key, project_name varchar, relation_name varchar, model_names varchar[], score float8);



create index onnx_prj_name_index on aisql.onnx_projects(project_name);



create table aisql.onnx_prj_models(project_id int, model_name varchar, algorithm varchar, model_data bytea, primary key (project_id, model_name));



create table aisql.nursery(parents varchar, has_nurs varchar, form varchar, children varchar, housing varchar, finance varchar, social varchar, health varchar, class varchar);

copy aisql.nursery from '/home/bigmath/plpython3u/brightml_core/brightml_core/data/nursery.data' (format csv, header false, delimiter ',', encoding 'utf-8');



create table aisql.flare_train(class char(1), largest_spot char(1), spot_distribution char(1), activity char(1), evolution char(1), flare_activity char(1), historically_complex char(1), complex_pass char(1), area char(1), spot_area char(1), C_class smallint, M_class smallint, X_class smallint);

copy aisql.flare_train from '/home/bigmath/plpython3u/brightml_core/brightml_core/data/flare.train' (format csv, header false, delimiter ' ', encoding 'utf-8');



create table aisql.flare_test(class char(1), largest_spot char(1), spot_distribution char(1), activity char(1), evolution char(1), flare_activity char(1), historically_complex char(1), complex_pass char(1), area char(1), spot_area char(1), C_class smallint, M_class smallint, X_class smallint);

copy aisql.flare_test from '/home/bigmath/plpython3u/brightml_core/brightml_core/data/flare.test' (format csv, header false, delimiter ' ', encoding 'utf-8');



create table aisql.uscensus_org(caseid int primary key,dAge smallint,dAncstry1 smallint,dAncstry2 smallint,iAvail smallint,iCitizen smallint,iClass smallint,dDepart smallint,iDisabl1 smallint,iDisabl2 smallint,iEnglish smallint,iFeb55 smallint,iFertil smallint,dHispanic smallint,dHour89 smallint,dHours smallint,iImmigr smallint,dIncome1 smallint,dIncome2 smallint,dIncome3 smallint,dIncome4 smallint,dIncome5 smallint,dIncome6 smallint,dIncome7 smallint,dIncome8 smallint,dIndustry smallint,iKorean smallint,iLang1 smallint,iLooking smallint,iMarital smallint,iMay75880 smallint,iMeans smallint,iMilitary smallint,iMobility smallint,iMobillim smallint,dOccup smallint,iOthrserv smallint,iPerscare smallint,dPOB smallint,dPoverty smallint,dPwgt1 smallint,iRagechld smallint,dRearning smallint,iRelat1 smallint,iRelat2 smallint,iRemplpar smallint,iRiders smallint,iRlabor smallint,iRownchld smallint,dRpincome smallint,iRPOB smallint,iRrelchld smallint,iRspouse smallint,iRvetserv smallint,iSchool smallint,iSept80 smallint,iSex smallint,iSubfam1 smallint,iSubfam2 smallint,iTmpabsnt smallint,dTravtime smallint,iVietnam smallint,dWeek89 smallint,iWork89 smallint,iWorklwk smallint,iWWII smallint,iYearsch smallint,iYearwrk smallint,dYrsserv smallint);

copy aisql.uscensus_org from '/home/bigmath/plpython3u/brightml_core/brightml_core/data/USCensus1990.data.txt' (format csv, header true, delimiter ',', encoding 'utf-8');



create view aisql.uscensus(caseid, data) as (select caseid, array[dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv] from aisql.uscensus_org);



select aisql.train_onnx(project_name=>'digits.xgboost.classification', task=>'classification', algorithm=>'xgboost', relation_name=>'digits', preprocess=>'{"min_max": {"feature_range": [-1, 1]}}');



select aisql.train_onnx(project_name=>'diabetes.xgboost.regression', task=>'regression', algorithm=>'xgboost', relation_name=>'diabetes', preprocess=>'{"standard": {}}');



select aisql.train_onnx(project_name=>'uscensus.kmeans.clustering', task=>'clustering', algorithm=>'kmeans', relation_name=>'aisql.uscensus');



select aisql.load_dataset('digits');

select * from aisql.digits limit 10;



select aisql.load_dataset('diabetes');

select * from aisql.diabetes limit 10;



select aisql.train(project_name=>'nursery.xgboost.classification', task=>'classification', algorithm=>'xgboost', relation_name=>'aisql.nursery', x_column_names=>string_to_array('parents,has_nurs,form,children,housing,finance,social,health', ','), y_column_name=>'class', preprocess=>'{"parents":{"ordinal":{},"min_max":{"feature_range": [-1, 1]}}, "has_nurs":{"ordinal":{},"impute":{"strategy":"constant","fill_value":0}}, "form":{"ordinal":{},"standard":{}}, "children":{"ordinal":{},"impute":{"strategy":"mean"}}, "housing":{"ordinal":{}}, "finance":{"one_hot":{}}, "social":{"ordinal":{}}, "health":{"ordinal":{}}, "class":{"encode":{}}}');



select aisql.train(project_name=>'nursery.catboost.classification', task=>'classification', algorithm=>'catboost', relation_name=>'aisql.nursery', x_column_names=>string_to_array('parents,has_nurs,form,children,housing,finance,social,health', ','), y_column_name=>'class', cat_columns=>array[0,1,2,3,4,5,6,7]);



select aisql.train(project_name=>'flare.xgboost.regression', task=>'regression', algorithm=>'xgboost', relation_name=>'aisql.flare_train', x_column_names=>string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), y_column_name=>'c_class', preprocess=>'{"_ALL_X_":{"ordinal":{}}}');



select aisql.train(project_name=>'flare.catboost.regression', task=>'regression', algorithm=>'catboost', relation_name=>'aisql.flare_train', x_column_names=>string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), y_column_name=>'c_class', cat_columns=>array[0,1,2,3,4,5,6,7,8,9]);



select aisql.train(project_name=>'uscensus.kmeans.clustering', task=>'clustering', algorithm=>'kmeans', relation_name=>'aisql.uscensus_org', x_column_names=>string_to_array('dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv', ','));



select class, aisql.predict('nursery.xgboost.classification', make_json_map(array[make_kv('parents', parents, TRUE), make_kv('has_nurs', has_nurs, TRUE), make_kv('form', form, TRUE), make_kv('children', children, TRUE), make_kv('housing', housing, TRUE), make_kv('finance', finance, TRUE), make_kv('social', social, TRUE), make_kv('health', health, TRUE)])) from aisql.nursery limit 10;



select class, aisql.predict('nursery.catboost.classification', make_json_map(array[make_kv('parents', parents, TRUE), make_kv('has_nurs', has_nurs, TRUE), make_kv('form', form, TRUE), make_kv('children', children, TRUE), make_kv('housing', housing, TRUE), make_kv('finance', finance, TRUE), make_kv('social', social, TRUE), make_kv('health', health, TRUE)])) from aisql.nursery limit 10;



select c_class, aisql.predict('flare.xgboost.regression', make_json_map(make_kv_array(string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), array[class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area], TRUE))) from aisql.flare_test where spot_area='1' limit 10;



select c_class, aisql.predict('flare.catboost.regression', make_json_map(make_kv_array(string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), array[class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area], TRUE))) from aisql.flare_test where spot_area='1' limit 10;



select aisql.predict('uscensus.kmeans.clustering', make_json_map(make_kv_array(string_to_array('dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv', ','), array[dAge::text,dAncstry1::text,dAncstry2::text,iAvail::text,iCitizen::text,iClass::text,dDepart::text,iDisabl1::text,iDisabl2::text,iEnglish::text,iFeb55::text,iFertil::text,dHispanic::text,dHour89::text,dHours::text,iImmigr::text,dIncome1::text,dIncome2::text,dIncome3::text,dIncome4::text,dIncome5::text,dIncome6::text,dIncome7::text,dIncome8::text,dIndustry::text,iKorean::text,iLang1::text,iLooking::text,iMarital::text,iMay75880::text,iMeans::text,iMilitary::text,iMobility::text,iMobillim::text,dOccup::text,iOthrserv::text,iPerscare::text,dPOB::text,dPoverty::text,dPwgt1::text,iRagechld::text,dRearning::text,iRelat1::text,iRelat2::text,iRemplpar::text,iRiders::text,iRlabor::text,iRownchld::text,dRpincome::text,iRPOB::text,iRrelchld::text,iRspouse::text,iRvetserv::text,iSchool::text,iSept80::text,iSex::text,iSubfam1::text,iSubfam2::text,iTmpabsnt::text,dTravtime::text,iVietnam::text,dWeek89::text,iWork89::text,iWorklwk::text,iWWII::text,iYearsch::text,iYearwrk::text,dYrsserv::text]))) from aisql.uscensus_org limit 10;



select unnest(array_agg(class)), unnest(aisql.batch_predict('nursery.xgboost.classification', make_json_map(make_kv_ext_array(string_to_array('parents,has_nurs,form,children,housing,finance,social,health', ','), array[array_agg(parents),array_agg(has_nurs),array_agg(form),array_agg(children),array_agg(housing),array_agg(finance),array_agg(social),array_agg(health)], TRUE)))) from aisql.nursery limit 10;



select unnest(array_agg(class)), unnest(aisql.batch_predict('nursery.catboost.classification', make_json_map(make_kv_ext_array(string_to_array('parents,has_nurs,form,children,housing,finance,social,health', ','), array[array_agg(parents),array_agg(has_nurs),array_agg(form),array_agg(children),array_agg(housing),array_agg(finance),array_agg(social),array_agg(health)], TRUE)))) from aisql.nursery limit 10;



select unnest(array_agg(c_class)), unnest(aisql.batch_predict('flare.xgboost.regression', make_json_map(make_kv_ext_array(string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), array[array_agg(class),array_agg(largest_spot),array_agg(spot_distribution),array_agg(activity),array_agg(evolution),array_agg(flare_activity),array_agg(historically_complex),array_agg(complex_pass),array_agg(area),array_agg(spot_area)], TRUE)))) from aisql.flare_test where spot_area='1' limit 10;



select unnest(array_agg(c_class)), unnest(aisql.batch_predict('flare.catboost.regression', make_json_map(make_kv_ext_array(string_to_array('class,largest_spot,spot_distribution,activity,evolution,flare_activity,historically_complex,complex_pass,area,spot_area', ','), array[array_agg(class),array_agg(largest_spot),array_agg(spot_distribution),array_agg(activity),array_agg(evolution),array_agg(flare_activity),array_agg(historically_complex),array_agg(complex_pass),array_agg(area),array_agg(spot_area)], TRUE)))) from aisql.flare_test where spot_area='1' limit 10;



select aisql.batch_predict('uscensus.kmeans.clustering', make_json_map(make_kv_ext_array(string_to_array('dAge,dAncstry1,dAncstry2,iAvail,iCitizen,iClass,dDepart,iDisabl1,iDisabl2,iEnglish,iFeb55,iFertil,dHispanic,dHour89,dHours,iImmigr,dIncome1,dIncome2,dIncome3,dIncome4,dIncome5,dIncome6,dIncome7,dIncome8,dIndustry,iKorean,iLang1,iLooking,iMarital,iMay75880,iMeans,iMilitary,iMobility,iMobillim,dOccup,iOthrserv,iPerscare,dPOB,dPoverty,dPwgt1,iRagechld,dRearning,iRelat1,iRelat2,iRemplpar,iRiders,iRlabor,iRownchld,dRpincome,iRPOB,iRrelchld,iRspouse,iRvetserv,iSchool,iSept80,iSex,iSubfam1,iSubfam2,iTmpabsnt,dTravtime,iVietnam,dWeek89,iWork89,iWorklwk,iWWII,iYearsch,iYearwrk,dYrsserv', ','), array[array_agg(dAge::text),array_agg(dAncstry1::text),array_agg(dAncstry2::text),array_agg(iAvail::text),array_agg(iCitizen::text),array_agg(iClass::text),array_agg(dDepart::text),array_agg(iDisabl1::text),array_agg(iDisabl2::text),array_agg(iEnglish::text),array_agg(iFeb55::text),array_agg(iFertil::text),array_agg(dHispanic::text),array_agg(dHour89::text),array_agg(dHours::text),array_agg(iImmigr::text),array_agg(dIncome1::text),array_agg(dIncome2::text),array_agg(dIncome3::text),array_agg(dIncome4::text),array_agg(dIncome5::text),array_agg(dIncome6::text),array_agg(dIncome7::text),array_agg(dIncome8::text),array_agg(dIndustry::text),array_agg(iKorean::text),array_agg(iLang1::text),array_agg(iLooking::text),array_agg(iMarital::text),array_agg(iMay75880::text),array_agg(iMeans::text),array_agg(iMilitary::text),array_agg(iMobility::text),array_agg(iMobillim::text),array_agg(dOccup::text),array_agg(iOthrserv::text),array_agg(iPerscare::text),array_agg(dPOB::text),array_agg(dPoverty::text),array_agg(dPwgt1::text),array_agg(iRagechld::text),array_agg(dRearning::text),array_agg(iRelat1::text),array_agg(iRelat2::text),array_agg(iRemplpar::text),array_agg(iRiders::text),array_agg(iRlabor::text),array_agg(iRownchld::text),array_agg(dRpincome::text),array_agg(iRPOB::text),array_agg(iRrelchld::text),array_agg(iRspouse::text),array_agg(iRvetserv::text),array_agg(iSchool::text),array_agg(iSept80::text),array_agg(iSex::text),array_agg(iSubfam1::text),array_agg(iSubfam2::text),array_agg(iTmpabsnt::text),array_agg(dTravtime::text),array_agg(iVietnam::text),array_agg(dWeek89::text),array_agg(iWork89::text),array_agg(iWorklwk::text),array_agg(iWWII::text),array_agg(iYearsch::text),array_agg(iYearwrk::text),array_agg(dYrsserv::text)]))) from aisql.uscensus_org where caseid<10010;



select label, aisql.predict_onnx('digits.xgboost.classification', data) from aisql.digits limit 10;



select label, aisql.predict_onnx('diabetes.xgboost.regression', data) from aisql.diabetes limit 10;



select aisql.predict_onnx('uscensus.kmeans.clustering', data) from aisql.uscensus limit 10;



select unnest(array_agg(label)), unnest(aisql.batch_predict_onnx('digits.xgboost.classification', array_agg(data))) from aisql.digits limit 10;



select unnest(array_agg(label)), unnest(aisql.batch_predict_onnx('diabetes.xgboost.regression', array_agg(data))) from aisql.diabetes limit 10;



select aisql.batch_predict_onnx('uscensus.kmeans.clustering', array_agg(data)) from aisql.uscensus where caseid<10010;