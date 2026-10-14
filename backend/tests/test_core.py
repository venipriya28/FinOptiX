import pandas as pd
import pytest
from app.core import clean_usage, recommendations, simulate, cost_summary, ranking_comparison, detect_anomalies, forecast, anomaly_evaluation
from app.core import FORECAST_FEATURES, ANOMALY_FEATURES, EVALUATION_LABELS
from app.generate_data import generate
from app.catalog import load_catalog, CatalogError
from app.simulator import simulate_options, SimulationError, risk_aware_score
import numpy as np
import copy

def sample():
    rows=[]
    for i in range(12):
        for day in range(40):
            rows.append(dict(timestamp=(pd.Timestamp('2026-08-01',tz='UTC')+pd.Timedelta(days=day)).isoformat(),cloud_provider="AWS",service="EC2",region="ap-south-1",resource_id=f"EC2-{i}",resource_type="vm",instance_type="m5.2xlarge",cpu_utilization=10 if i==0 else 55,memory_utilization=18 if i==0 else 60,network_usage=10,storage_usage=20,running_hours=10,monthly_cost=21400,department="Eng",project="Atlas",environment="production"))
    return clean_usage(pd.DataFrame(rows))

def test_pipeline_deduplicates_and_derives_features():
    df=sample(); assert len(df)==480 and "resource_efficiency_score" in df and df.iloc[0].daily_cost > 0
    assert {'weekly_cost','7_day_avg_cost','30_day_avg_cost','cost_growth_rate'} <= set(df.columns)

def test_summary_uses_current_resource_monthly_cost_once():
    assert cost_summary(sample())["total_monthly_cost"] == 12*21400

def test_rules_explain_and_rank_low_utilization():
    recs=recommendations(sample()); assert recs[0]["resource_id"]=="EC2-0" and recs[0]["evidence"]["cpu_pct"]==10

def priced_history(cpu=10, memory=20, runtime=24, days=90, environment="development", service="EC2", sku="m5.2xlarge", storage_use=20):
    cat=load_catalog(); provider=cat['providers']['AWS']; spec=provider['compute_skus'][sku]; rows=[]
    rng=np.random.default_rng(11)
    for day in range(days):
        c=float(np.clip(rng.normal(cpu, 2), 0, 99)); m=float(np.clip(rng.normal(memory, 2), 0, 99)); h=float(np.clip(rng.normal(runtime, .25), 0, 24))
        util=(c+m)/200; compute=30*h*spec['hourly_inr']*provider['regions']['ap-south-1']*(cat['billing']['utilization_cost_multiplier']['base']+cat['billing']['utilization_cost_multiplier']['per_utilization_fraction']*util)
        used=storage_use/100; storage=1000*provider['storage_inr_per_gb_month']*(1+cat['billing']['storage_io_surcharge_per_usage_fraction']*used)
        net=max(0,25*30-provider['network_free_gb_per_month'])*provider['network_inr_per_gb']; rows.append(dict(timestamp=f"2026-07-{1:02d}T00:00:00Z" if day==0 else (pd.Timestamp('2026-07-01',tz='UTC')+pd.Timedelta(days=day)).isoformat(),cloud_provider='AWS',service=service,region='ap-south-1',resource_id='EC2-TEST',resource_type='virtual_machine',instance_type=sku,cpu_utilization=c,memory_utilization=m,network_usage=25,storage_usage=storage_use,allocated_storage_gb=1000,running_hours=h,monthly_cost=compute+storage+net,compute_monthly_cost=compute,storage_monthly_cost=storage,network_monthly_cost=net,department='Engineering',project='Atlas',environment=environment))
    return clean_usage(pd.DataFrame(rows))

def test_catalog_priced_simulator_compares_feasible_alternatives():
    result=simulate_options(priced_history(), 'EC2-TEST')
    rightsizes=[x for x in result['alternatives'] if x['action']=='instance_rightsizing']
    assert {x['target_instance'] for x in rightsizes}=={'m5.xlarge'}
    assert any(x.get('target_instance')=='t3.medium' and x.get('performance_constraint_violations') for x in result['rejected_options'])
    assert all(x['monthly_saving'] > 0 and x['annualized_saving']==round(x['monthly_saving']*12,2) for x in rightsizes)
    assert all(x['expected_cpu_p95_pct'] > x['evidence']['cpu_p95_pct'] for x in rightsizes)
    assert all('illustrative_assumption' in x['assumptions'][0] for x in rightsizes)

def test_provider_scoped_resource_identity_prevents_cross_cloud_mixing():
    from app.simulator import SimulationError
    aws=priced_history(days=40)
    azure=aws.copy()
    azure['cloud_provider']='Azure'; azure['service']='Azure VM'; azure['region']='centralindia'; azure['instance_type']='Standard_D8s_v5'
    combined=pd.concat([aws,azure],ignore_index=True)
    with pytest.raises(SimulationError,match='ambiguous'):
        simulate_options(combined,'EC2-TEST')
    assert simulate_options(combined,'EC2-TEST',cloud_provider='AWS')['cloud_provider']=='AWS'

def test_downsize_rejects_peak_cpu_capacity_violation():
    result=simulate_options(priced_history(cpu=70,memory=70), 'EC2-TEST')
    assert not any(x['action']=='instance_rightsizing' for x in result['alternatives'])
    assert any(x.get('action')=='instance_rightsizing' and x.get('performance_constraint_violations') for x in result['rejected_options'])

def test_insufficient_history_is_explicit_and_not_ranked():
    result=simulate_options(priced_history(days=12), 'EC2-TEST')
    assert result['data_sufficient'] is False and 'Insufficient evidence' in result['warning']
    assert not result['alternatives']

def test_missing_utilization_marks_evidence_insufficient():
    raw=priced_history(days=45).drop(columns=['daily_cost','utilization_score','idle_hours','cost_per_usage','resource_efficiency_score','day','weekday','month'])
    raw.loc[raw.index[:12], 'cpu_utilization']=np.nan
    frame=clean_usage(raw)
    result=simulate_options(frame,'EC2-TEST')
    assert result['data_sufficient'] is False
    assert not result['alternatives']

def test_shutdown_respects_environment_runtime_window_and_minimum():
    eligible=simulate_options(priced_history(runtime=20,environment='development'), 'EC2-TEST')
    option=next(x for x in eligible['alternatives'] if x['action']=='scheduled_shutdown')
    assert option['action_details']['eligible_hours_per_day'] <= 8
    rejected=simulate_options(priced_history(runtime=10,environment='production'), 'EC2-TEST')
    assert not any(x['action']=='scheduled_shutdown' for x in rejected['alternatives'])

def test_storage_rightsizing_uses_p99_and_capacity_headroom():
    result=simulate_options(priced_history(storage_use=12), 'EC2-TEST')
    option=next(x for x in result['alternatives'] if x['action']=='storage_optimization')
    assert option['target_instance'].endswith('GB') and option['monthly_saving']>0

def test_invalid_sku_configuration_is_rejected():
    frame=priced_history(); frame.loc[:,'instance_type']='bogus'
    with pytest.raises(CatalogError): simulate_options(frame,'EC2-TEST')

def test_zero_cost_and_empty_resource_produce_no_false_savings():
    frame=priced_history(days=40)
    for col in ['monthly_cost','compute_monthly_cost','storage_monthly_cost','network_monthly_cost']:
        frame[col]=0.0
    result=simulate_options(frame,'EC2-TEST')
    assert not result['alternatives']
    with pytest.raises(KeyError): simulate_options(frame,'missing')

def test_small_high_utilization_resource_has_no_feasible_rightsize():
    frame=priced_history(cpu=80,memory=85,sku='t3.medium',days=90).drop(columns=['allocated_storage_gb'])
    result=simulate_options(frame,'EC2-TEST')
    assert not result['alternatives']
    assert result['rejected_options']

def test_variable_workload_increases_risk_score():
    from app.simulator import _risk
    base={'cpu_cv':.05,'memory_cv':.05}
    variable={'cpu_cv':.8,'memory_cv':.7}
    calm=_risk(40,45,base,'development',load_catalog())[0]
    volatile=_risk(40,45,variable,'development',load_catalog())[0]
    assert volatile > calm

def test_negative_prices_are_rejected_by_catalog_validation():
    catalog=copy.deepcopy(load_catalog())
    catalog['providers']['AWS']['compute_skus']['m5.xlarge']['hourly_inr']=-1
    # A fresh catalog is validated at load time; dynamic caller configs are validated before simulation too.
    with pytest.raises((CatalogError, SimulationError, ValueError)):
        frame=priced_history()
        catalog['providers']['AWS']['compute_skus']['m5.xlarge']['hourly_inr']=-1
        simulate_options(frame,'EC2-TEST',catalog=catalog)

def test_risk_aware_ranking_compares_identical_feasible_pairs():
    comparison=ranking_comparison(priced_history())
    assert comparison['same_candidate_set'] is True
    assert {x['option_id'] for x in comparison['savings_only']}=={x['option_id'] for x in comparison['risk_aware']}

def test_risk_aware_objective_can_prefer_moderate_safer_saving():
    high_saving_high_risk=risk_aware_score(900,1000,.9,80)
    moderate_saving_low_risk=risk_aware_score(450,1000,.8,20)
    assert moderate_saving_low_risk > high_saving_high_risk

def test_ranking_evidence_term_excludes_factors_already_present_in_risk():
    from app.simulator import _ranking_evidence_quality
    factors={'observation_volume':.8,'coverage':.9,'stability':.1,'recommendation_consistency':.2,'peak_metrics':1,'price_reliability':.55}
    baseline=_ranking_evidence_quality(factors,load_catalog())
    factors['stability']=1; factors['recommendation_consistency']=1
    assert _ranking_evidence_quality(factors,load_catalog())==baseline

def test_ranking_rejects_negative_savings_and_invalid_confidence():
    with pytest.raises(SimulationError): risk_aware_score(-1,100,.9,10)
    with pytest.raises(SimulationError): risk_aware_score(10,100,1.5,10)

def test_end_to_end_observations_to_explained_simulation_and_savings():
    frame=priced_history(days=90,storage_use=12)
    # Inject an observed recent event, then use the same processed rows for detection and simulation.
    frame.loc[frame.index[-1], 'monthly_cost'] *= 3
    frame.loc[frame.index[-1], 'network_usage'] *= 3
    anomalies=detect_anomalies(frame)
    result=simulate_options(frame,'EC2-TEST')
    assert result['current']['observations']==90
    assert result['alternatives']
    chosen=result['alternatives'][0]
    assert chosen['monthly_saving']>0 and chosen['annualized_saving']==round(chosen['monthly_saving']*12,2)
    assert chosen['assumptions'] and chosen['confidence_interpretation'].startswith('Evidence quality')
    assert all(option['feasible'] for option in result['alternatives'])
    assert isinstance(anomalies,list)

def test_root_cause_compares_measured_windows_and_does_not_infer_unit_prices():
    from app.core import _root_cause_evidence
    frame=priced_history(days=90)
    at=pd.to_datetime(frame.timestamp.max(),utc=True)
    analysis=_root_cause_evidence(frame,at)
    assert analysis['status']=='measured_comparison'
    assert analysis['baseline_observed_days']==30 and analysis['recent_observed_days']==7
    assert analysis['resource_count_before']==analysis['resource_count_recent']==1
    assert analysis['service_region_contributions'][0]['service']=='EC2'
    assert 'not measured' in analysis['unit_price_change']
    assert not analysis['resources_not_observed_in_baseline']
    assert 'does not establish their creation dates' in analysis['resource_presence_note']

def test_validation_rejects_invalid_utilization():
    frame=pd.DataFrame([dict(timestamp="2026-09-01",cloud_provider="AWS",service="EC2",region="x",resource_id="x",resource_type="vm",instance_type="t",cpu_utilization=150,memory_utilization=1,network_usage=1,storage_usage=1,running_hours=1,monthly_cost=1,department="x",project="x",environment="x")])
    with pytest.raises(ValueError): clean_usage(frame)

def test_empty_data_and_invalid_runtime_are_rejected():
    with pytest.raises(ValueError,match='No usage observations'):
        clean_usage(pd.DataFrame(columns=['timestamp','cloud_provider','service','region','resource_id','resource_type','instance_type','cpu_utilization','memory_utilization','network_usage','storage_usage','running_hours','monthly_cost','department','project','environment']))
    raw=priced_history(days=40).drop(columns=['daily_cost','weekly_cost','7_day_avg_cost','30_day_avg_cost','monthly_cost_30d','cost_growth_rate','utilization_score','idle_hours','cost_per_usage','resource_efficiency_score','day','weekday','month'])
    raw.loc[raw.index[0],'running_hours']=25
    with pytest.raises(ValueError,match='Running hours'):
        clean_usage(raw)

def test_seeded_generator_covers_scenarios_and_labels_are_not_model_features():
    one = generate(seed=42, days=180, resources=10)
    two = generate(seed=42, days=180, resources=10)
    pd.testing.assert_frame_equal(one, two)
    assert set(one.scenario_type) == {"normal", "underutilized", "idle", "over_provisioned", "highly_utilized", "sudden_cost_spike", "gradual_cost_increase", "seasonal_usage", "newly_deployed", "occasional_peak"}
    assert all(column not in FORECAST_FEATURES + ANOMALY_FEATURES for column in EVALUATION_LABELS)
    assert one.is_anomaly.sum() > 0 and one.optimization_opportunity.sum() > 0

def test_cost_is_usage_correlated_and_new_resource_history_starts_at_launch():
    frame=generate(seed=12, days=180, resources=10)
    idle=frame[frame.scenario_type=="idle"]
    high=frame[frame.scenario_type=="highly_utilized"]
    assert idle.running_hours.mean() < high.running_hours.mean()
    assert idle.monthly_cost.mean() < high.monthly_cost.mean()
    new=frame[frame.scenario_type=="newly_deployed"]
    assert len(new) < 180 and new.timestamp.min() > frame.timestamp.min()

def test_forecast_selects_from_validation_and_reports_untouched_time_holdout():
    frame=clean_usage(generate(seed=2026,days=180,resources=20,scenario_offset=4))
    result=forecast(frame)
    assert result["selection"]["test_window_is_not_used_for_selection"] is True
    assert result["selection"]["method"].startswith("lowest MAE on a chronological validation")
    assert len(result["model_comparison"]) == 3
    assert {"mae","rmse","mape_pct","r2"} <= result["metrics"].keys()
    test_rows=[row for row in result["series"] if row["predicted"] is not None]
    assert test_rows and all(row["actual"] >= 0 and row["predicted"] >= 0 for row in test_rows)
    assert result["next_day"] >= 0 and result["next_week"] >= 0 and result["next_month"] >= 0

def test_short_history_forecast_is_explicitly_limited():
    raw=generate(seed=42,days=180,resources=10)
    frame=clean_usage(raw[pd.to_datetime(raw.timestamp,utc=True) < pd.Timestamp(raw.timestamp.min()) + pd.Timedelta(days=30)])
    result=forecast(frame)
    assert result["metrics"] is None
    assert result["model_comparison"] == []
    assert "35 daily observations" in result["warning"]

def test_anomaly_hybrid_is_measured_and_operational_detector_excludes_labels():
    frame=clean_usage(generate(seed=106755,days=180,resources=20,scenario_offset=4))
    metrics=anomaly_evaluation(frame)
    assert "hybrid_iforest_or_rolling_mad" in metrics
    assert metrics["hybrid_iforest_or_rolling_mad"]["predicted_rows"] > 0
    # Removing evaluation labels must not change the operational anomaly signals.
    without_labels=frame.drop(columns=[column for column in EVALUATION_LABELS if column in frame])
    with_labels=detect_anomalies(frame)
    no_labels=detect_anomalies(without_labels)
    assert [(x["resource_id"],x["timestamp"]) for x in with_labels] == [(x["resource_id"],x["timestamp"]) for x in no_labels]
