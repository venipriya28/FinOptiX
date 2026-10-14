from fastapi.testclient import TestClient
from app import main
from app.main import app
from app.generate_data import generate

client = TestClient(app)

def test_health_and_analytics_endpoints():
    root = client.get('/')
    assert root.status_code == 200 and root.json()['docs'] == '/docs'
    assert client.get('/health').status_code == 200
    assert client.get('/cost-summary').json()['resource_count'] > 0
    assert isinstance(client.get('/recommendations').json(), list)

def test_forecast_endpoint_exposes_validation_selected_candidates():
    response=client.get('/forecast')
    assert response.status_code==200
    payload=response.json()
    assert payload['selection']['test_window_is_not_used_for_selection'] is True
    assert len(payload['model_comparison'])==3
    assert payload['metrics']['r2'] is not None

def test_unconfigured_lan_vite_origin_is_not_allowed():
    response = client.options('/cost-summary', headers={
        'Origin': 'http://10.71.34.143:5174',
        'Access-Control-Request-Method': 'GET',
    })
    assert response.status_code == 400
    assert 'access-control-allow-origin' not in response.headers

def test_what_if_validates_missing_resource():
    response = client.post('/what-if', json={'resource_id':'absent-resource','action':'compare'})
    assert response.status_code == 404

def test_what_if_returns_multiple_catalog_alternatives():
    response=client.post('/what-if',json={'resource_id':'AZURE-101','action':'compare'})
    assert response.status_code==200
    payload=response.json()
    assert 'current' in payload and 'alternatives' in payload and 'rejected_options' in payload
    assert payload['pricing_source']=='illustrative_assumption'

def test_anomaly_evaluation_uses_planted_labels_after_prediction():
    result=client.get('/anomaly-evaluation').json()
    assert {'precision','recall','f1','ground_truth_positive_rows'} <= result.keys()
    assert 'excluded from model inputs' in result['label_source']
    assert result['ground_truth_positive_rows'] > 0 and result['holdout_rows'] > 0

def test_chat_returns_grounded_answer():
    response = client.post('/chat', json={'question':'How much can we save?'})
    assert response.status_code == 200 and response.json()['grounding']
    assert response.json()['mode'] == 'application_analytics'

def test_chat_knowledge_mode_retrieves_source_references(monkeypatch):
    monkeypatch.delenv('FINOPTIX_LLM_API_KEY',raising=False)
    monkeypatch.delenv('FINOPTIX_LLM_MODEL',raising=False)
    response=client.post('/chat',json={'question':'How should cloud resources be rightsized?', 'mode':'knowledge'})
    assert response.status_code==200
    payload=response.json()
    assert payload['mode']=='knowledge_retrieval' and payload['retrieved']>0
    assert payload['sources'][0]['reference'].startswith('http')

def test_knowledge_sources_endpoint_is_populated():
    sources=client.get('/knowledge/sources').json()
    assert len(sources)>=4
    assert all(source['reference'].startswith('http') for source in sources)

def test_end_to_end_csv_ingestion_to_simulation_and_savings(tmp_path):
    old_path, old_upload, old_frame, old_source = main.DATA, main.UPLOAD_DATA, main._df, main._active_source
    main.DATA, main.UPLOAD_DATA, main._df = tmp_path/'demo.csv', tmp_path/'uploaded.csv', None
    try:
        synthetic=generate(seed=73,days=180,resources=10)
        response=client.post('/upload',files={'file':('demo.csv',synthetic.to_csv(index=False),'text/csv')})
        assert response.status_code==200 and response.json()['loaded_rows']>0
        assert client.get('/cost-summary').json()['resource_count']==10
        recs=client.get('/recommendations').json()
        assert recs
        result=client.post('/what-if',json={'resource_id':recs[0]['resource_id'],'action':'compare'}).json()
        assert result['alternatives']
        assert all(option['monthly_saving']>0 for option in result['alternatives'])
        assert client.get('/savings').json()['monthly_opportunity']>0
    finally:
        main.DATA, main.UPLOAD_DATA, main._df, main._active_source = old_path, old_upload, old_frame, old_source
