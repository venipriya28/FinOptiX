-- PostgreSQL production schema. Synthetic demo runs locally without requiring a DB.
CREATE TABLE users (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), email TEXT UNIQUE NOT NULL, role TEXT NOT NULL DEFAULT 'viewer', created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE cloud_resources (id BIGSERIAL PRIMARY KEY, provider TEXT NOT NULL, resource_id TEXT NOT NULL, service TEXT NOT NULL, region TEXT NOT NULL, resource_type TEXT NOT NULL, instance_type TEXT, department TEXT, project TEXT, environment TEXT, UNIQUE(provider, resource_id));
CREATE TABLE cloud_usage (id BIGSERIAL PRIMARY KEY, resource_fk BIGINT NOT NULL REFERENCES cloud_resources(id) ON DELETE CASCADE, observed_at TIMESTAMPTZ NOT NULL, cpu_pct NUMERIC(5,2), memory_pct NUMERIC(5,2), network_usage NUMERIC, storage_usage NUMERIC, running_hours NUMERIC(5,2), utilization_score NUMERIC(5,2), idle_hours NUMERIC(6,2), raw_record JSONB NOT NULL DEFAULT '{}');
CREATE TABLE cloud_costs (id BIGSERIAL PRIMARY KEY, resource_fk BIGINT NOT NULL REFERENCES cloud_resources(id) ON DELETE CASCADE, period_start DATE NOT NULL, period_end DATE NOT NULL, amount NUMERIC(14,2) NOT NULL, currency CHAR(3) NOT NULL DEFAULT 'INR', source TEXT NOT NULL DEFAULT 'synthetic', UNIQUE(resource_fk,period_start,period_end,source));
CREATE TABLE anomalies (id BIGSERIAL PRIMARY KEY, resource_fk BIGINT REFERENCES cloud_resources(id), detected_at TIMESTAMPTZ NOT NULL DEFAULT now(), detector TEXT NOT NULL, severity TEXT NOT NULL, cost_impact NUMERIC(14,2), cause TEXT NOT NULL, evidence JSONB NOT NULL DEFAULT '{}');
CREATE TABLE recommendations (id BIGSERIAL PRIMARY KEY, resource_fk BIGINT NOT NULL REFERENCES cloud_resources(id), action TEXT NOT NULL, reason TEXT NOT NULL, evidence JSONB NOT NULL, monthly_saving NUMERIC(14,2), confidence NUMERIC(4,3), risk TEXT NOT NULL, priority_score NUMERIC(5,2), status TEXT NOT NULL DEFAULT 'proposed', created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE optimization_simulations (id BIGSERIAL PRIMARY KEY, recommendation_fk BIGINT REFERENCES recommendations(id), resource_fk BIGINT NOT NULL REFERENCES cloud_resources(id), action TEXT NOT NULL, before_cost NUMERIC(14,2) NOT NULL, predicted_cost NUMERIC(14,2) NOT NULL, risk TEXT NOT NULL, confidence NUMERIC(4,3), assumptions JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE savings (id BIGSERIAL PRIMARY KEY, simulation_fk BIGINT REFERENCES optimization_simulations(id), realized_month DATE NOT NULL, predicted_amount NUMERIC(14,2), realized_amount NUMERIC(14,2), currency CHAR(3) NOT NULL DEFAULT 'INR');
CREATE TABLE model_predictions (id BIGSERIAL PRIMARY KEY, model_name TEXT NOT NULL, target_date DATE NOT NULL, predicted_cost NUMERIC(14,2) NOT NULL, actual_cost NUMERIC(14,2), metrics JSONB, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE audit_logs (id BIGSERIAL PRIMARY KEY, user_fk UUID REFERENCES users(id), action TEXT NOT NULL, entity_type TEXT, entity_id TEXT, details JSONB NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE INDEX idx_usage_resource_time ON cloud_usage(resource_fk, observed_at DESC);
CREATE INDEX idx_cost_period ON cloud_costs(period_start, period_end);
CREATE INDEX idx_resource_dims ON cloud_resources(provider, service, region, department);
CREATE INDEX idx_recommendations_rank ON recommendations(status, priority_score DESC);
CREATE INDEX idx_anomalies_detected ON anomalies(detected_at DESC, severity);
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_resource_observed ON cloud_usage(resource_fk, observed_at);
CREATE INDEX IF NOT EXISTS idx_usage_observed_at ON cloud_usage(observed_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_anomaly_resource_time_detector ON anomalies(resource_fk,detected_at,detector);
CREATE UNIQUE INDEX IF NOT EXISTS uq_recommendation_resource_action_status ON recommendations(resource_fk,action,status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_prediction_model_date ON model_predictions(model_name,target_date);
-- Common query: monthly spend by service and department
-- SELECT r.service,r.department,sum(c.amount) FROM cloud_costs c JOIN cloud_resources r ON r.id=c.resource_fk
-- WHERE c.period_start >= date_trunc('month',current_date)::date GROUP BY 1,2 ORDER BY 3 DESC;
