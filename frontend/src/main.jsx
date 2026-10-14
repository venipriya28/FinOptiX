import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3, Bot, Check,
  ChevronDown, ChevronRight, CircleDollarSign, Cloud, Database, Download, FileUp, Gauge, HelpCircle,
  Layers3, Menu, MessageCircle, Play, RefreshCw, Search, ShieldCheck, Sparkles,
  TrendingUp, UploadCloud, X,
} from 'lucide-react';
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import './style.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const NAV = [
  { id: 'overview', label: 'Overview', icon: Layers3, group: 'WORKSPACE' },
  { id: 'costs', label: 'Cost explorer', icon: BarChart3, group: 'WORKSPACE' },
  { id: 'forecast', label: 'Forecast', icon: TrendingUp, group: 'WORKSPACE' },
  { id: 'anomalies', label: 'Anomalies & causes', icon: AlertTriangle, group: 'INSIGHTS' },
  { id: 'optimizations', label: 'Recommendations', icon: Sparkles, group: 'INSIGHTS' },
  { id: 'simulator', label: 'What-if simulator', icon: Activity, group: 'INSIGHTS' },
  { id: 'savings', label: 'Savings analysis', icon: CircleDollarSign, group: 'INSIGHTS' },
  { id: 'copilot', label: 'FinOps copilot', icon: Bot, group: 'ASSISTANT' },
];
const money = (n, digits = 0) => `₹${Number(n || 0).toLocaleString('en-IN', { maximumFractionDigits: digits })}`;
const compactMoney = (n) => `₹${new Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(Number(n || 0))}`;
const pct = (n, digits = 0) => `${Number(n || 0).toFixed(digits)}%`;
const dateLabel = (value) => value ? new Date(`${String(value).slice(0, 10)}T00:00:00`).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }) : '—';
const resourceKey = (row) => `${row.cloud_provider || 'unknown'}::${row.resource_id}`;

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  let payload;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function App() {
  const [page, setPage] = useState('overview');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const load = useCallback(async (quiet = false) => {
    if (quiet) setRefreshing(true); else setLoading(true);
    setError('');
    try {
      const [health, summary, trends, forecast, anomalies, recommendations, savings, ranking] = await Promise.all([
        request('/health'), request('/cost-summary'), request('/cost-trends'), request('/forecast'),
        request('/anomalies'), request('/recommendations'), request('/savings'), request('/optimization-ranking'),
      ]);
      setData({ health, summary, trends: Array.isArray(trends) ? trends : trends.data || [], forecast, anomalies: Array.isArray(anomalies) ? anomalies : anomalies.data || [], recommendations: Array.isArray(recommendations) ? recommendations : recommendations.data || [], savings, ranking });
    } catch (err) { setData(null); setError(err.message || 'Could not load FinOptiX data.'); }
    finally { setLoading(false); setRefreshing(false); }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const handleHash = () => { const next = window.location.hash.slice(1); if (NAV.some(item => item.id === next)) setPage(next); };
    window.addEventListener('hashchange', handleHash); handleHash();
    return () => window.removeEventListener('hashchange', handleHash);
  }, []);

  const go = (id) => { setPage(id); setMobileOpen(false); window.location.hash = id; window.scrollTo({ top: 0, behavior: 'smooth' }); };
  const current = NAV.find(item => item.id === page) || NAV[0];

  if (loading) return <div className="startup"><div className="startup-logo"><Cloud size={24} /></div><strong>FinOptiX</strong><span>Connecting to your FinOps workspace…</span><div className="loader-line" /></div>;
  if (error && !data) return <div className="startup"><div className="startup-logo error-icon"><AlertTriangle size={23} /></div><strong>Can’t reach FinOptiX API</strong><span>{error}</span><p>Start the backend in a terminal with <code>python -m uvicorn app.main:app --reload</code>, then try again.</p><button className="button primary" onClick={() => load()}><RefreshCw size={15} /> Retry connection</button></div>;

  return <div className="app-shell">
    {mobileOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)} />}
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Cloud size={20} /></div><div><b>FinOptiX</b><small>CLOUD COST INTELLIGENCE</small></div><button className="mobile-close" onClick={() => setMobileOpen(false)} aria-label="Close menu"><X size={17} /></button></div>
      <div className="tenant"><span className="status-dot" /> {data.health.data_source || 'API DATA'} <ChevronDown size={13} /></div>
      {['WORKSPACE', 'INSIGHTS', 'ASSISTANT'].map(group => <div className="nav-group" key={group}><div className="nav-label">{group}</div>{NAV.filter(item => item.group === group).map(item => {
        const Icon = item.icon;
        return <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => go(item.id)}><Icon size={17} strokeWidth={1.8} /><span>{item.label}</span>{page === item.id && <span className="nav-active-dot" />}</button>;
      })}</div>)}
      <div className="sidebar-bottom"><div className="demo-avatar">FX</div><div><b>FinOps workspace</b><small>Local demonstration</small></div><HelpCircle size={16} className="help-icon" /></div>
    </aside>
    <main className="main-area">
      <header className="topbar"><div className="topbar-left"><button className="mobile-menu" aria-label="Open menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button><div className="breadcrumbs">FinOptiX <ChevronRight size={14} /> <span>{current.label}</span></div></div><div className="topbar-actions"><div className="live-indicator"><span /> API connected</div><button className="icon-button" title="Refresh data" onClick={() => load(true)} disabled={refreshing}><RefreshCw size={16} className={refreshing ? 'spinning' : ''} /></button><div className="avatar-small">F</div></div></header>
      <div className="page-wrap">
        {error && <div className="inline-warning"><AlertTriangle size={15} /> Some data could not refresh: {error}</div>}
        {page === 'overview' && <Overview data={data} go={go} />}
        {page === 'costs' && <CostExplorer data={data} />}
        {page === 'forecast' && <ForecastPage forecast={data.forecast} trends={data.trends} />}
        {page === 'anomalies' && <AnomaliesPage anomalies={data.anomalies} />}
        {page === 'optimizations' && <RecommendationsPage recommendations={data.recommendations} ranking={data.ranking} go={go} />}
        {page === 'simulator' && <SimulatorPage recommendations={data.recommendations} />}
        {page === 'savings' && <SavingsPage recommendations={data.recommendations} savings={data.savings} />}
        {page === 'copilot' && <CopilotPage />}
        <footer className="site-footer"><span>FinOptiX · Explainable risk-aware cloud cost optimization</span><span>ILLUSTRATIVE CATALOG PRICING · NO CLOUD CHANGES EXECUTED</span></footer>
      </div>
    </main>
  </div>;
}

function PageHeading({ eyebrow, title, subtitle, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>{action}</div>;
}
function Panel({ className = '', children, ...props }) { return <section className={`panel ${className}`} {...props}>{children}</section>; }
function PanelHeading({ title, description, action }) { return <div className="panel-heading"><div><h2>{title}</h2>{description && <p>{description}</p>}</div>{action}</div>; }
function Metric({ label, value, detail, icon: Icon, accent = 'green' }) {
  return <Panel className="metric-card"><div className={`metric-icon ${accent}`}>{Icon && <Icon size={17} />}</div><div className="metric-label">{label}</div><div className="metric-value">{value}</div><div className="metric-detail">{detail}</div></Panel>;
}
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="chart-tooltip"><b>{dateLabel(label)}</b>{payload.map(row => <div key={row.dataKey}><span style={{ color: row.color }}>{row.name || row.dataKey}</span><strong>{money(row.value)}</strong></div>)}</div>;
}
function RiskTag({ risk }) { return <span className={`risk-tag ${(risk || 'medium').toLowerCase()}`}>{risk || 'MEDIUM'}</span>; }

function Overview({ data, go }) {
  const { summary, recommendations, anomalies, forecast, trends } = data;
  const monthlySpend = summary.total_monthly_cost || 0;
  const totalSaving = recommendations.reduce((sum, item) => sum + Number(item.estimated_monthly_saving || 0), 0);
  const monthTrend = trends.slice(-90);
  const serviceData = Object.entries(summary.by_service || {}).map(([name, cost]) => ({ name, cost: Number(cost) })).slice(0, 7);
  const lowRisk = recommendations.filter(row => row.risk === 'LOW').length;
  return <>
    <PageHeading eyebrow="FINOPS / EXECUTIVE OVERVIEW" title="Cloud cost intelligence" subtitle="Understand spend, find opportunities, and test changes before they reach production." action={<button className="button secondary" onClick={() => go('simulator')}><Activity size={15} /> Open simulator</button>} />
    <div className="callout"><div className="callout-icon"><Sparkles size={18} /></div><div><b>Risk-aware optimization is ready</b><span>Recommendations are ranked using savings, confidence, and operational risk.</span></div><button className="text-button" onClick={() => go('optimizations')}>Review opportunities <ChevronRight size={15} /></button></div>
    <div className="metric-grid">
      <Metric label="MONTHLY CLOUD SPEND" value={money(monthlySpend)} detail="Latest resource monthly run-rate" icon={Cloud} />
      <Metric label="POTENTIAL SAVINGS" value={money(totalSaving)} detail={`Per month · ${recommendations.length} opportunities`} icon={ArrowDownRight} accent="blue" />
      <Metric label="RESOURCES MONITORED" value={summary.resource_count || 0} detail={`${Object.keys(summary.by_provider || {}).length} cloud provider styles`} icon={Database} accent="violet" />
      <Metric label="ANOMALIES TO REVIEW" value={anomalies.length} detail={`${lowRisk} low-risk actions available`} icon={ShieldCheck} accent="amber" />
    </div>
    <div className="content-grid overview-grid">
      <Panel><PanelHeading title="Spend over time" description="Daily modeled cloud cost · last 90 days" action={<span className="subtle-chip"><span className="status-dot" /> All providers</span>} /><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><AreaChart data={monthTrend} margin={{ top: 12, right: 10, left: 0, bottom: 0 }}><defs><linearGradient id="spendFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#2b9a77" stopOpacity={0.23} /><stop offset="100%" stopColor="#2b9a77" stopOpacity={0.015} /></linearGradient></defs><CartesianGrid stroke="#edf1ef" vertical={false} strokeDasharray="4 5" /><XAxis dataKey="date" tickFormatter={dateLabel} tickLine={false} axisLine={false} minTickGap={34} /><YAxis tickFormatter={compactMoney} tickLine={false} axisLine={false} width={55} /><Tooltip content={<ChartTooltip />} /><Area type="monotone" dataKey="cost" name="Daily cost" stroke="#238664" strokeWidth={2.3} fill="url(#spendFill)" activeDot={{ r: 4 }} /></AreaChart></ResponsiveContainer></div></Panel>
      <Panel className="forecast-summary"><PanelHeading title="Cost forecast" description={forecast.model || 'Forecast outlook'} action={<TrendingUp size={18} className="accent-icon" />} /><div className="forecast-main">{money(forecast.next_month)}<span>next 30 days</span></div><div className="forecast-line"><span>Tomorrow</span><b>{money(forecast.next_day)}</b></div><div className="forecast-line"><span>Next 7 days</span><b>{money(forecast.next_week)}</b></div><div className="forecast-quality">{forecast.metrics ? `Holdout MAE ${money(forecast.metrics.mae)} · MAPE ${pct(forecast.metrics.mape_pct, 2)}` : 'Baseline forecast · metrics unavailable'}</div><button className="text-button small" onClick={() => go('forecast')}>View forecast details <ChevronRight size={14} /></button></Panel>
    </div>
    <div className="content-grid overview-bottom">
      <Panel><PanelHeading title="Top optimization opportunities" description="Risk and confidence are shown for every action." action={<button className="text-button" onClick={() => go('optimizations')}>View all <ChevronRight size={14} /></button>} /><RecommendationTable recommendations={recommendations.slice(0, 5)} compact onSelect={() => go('simulator')} /></Panel>
      <Panel><PanelHeading title="Spend by service" description="Latest monthly resource totals" /><div className="small-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={serviceData} layout="vertical" margin={{ left: 0, right: 12 }}><CartesianGrid stroke="#edf1ef" horizontal={false} /><XAxis type="number" tickFormatter={compactMoney} tickLine={false} axisLine={false} /><YAxis dataKey="name" type="category" width={90} tickLine={false} axisLine={false} /><Tooltip formatter={v => money(v)} /><Bar dataKey="cost" fill="#56a990" radius={[0, 5, 5, 0]} barSize={15} /></BarChart></ResponsiveContainer></div></Panel>
    </div>
    <div className="content-grid one-col"><Panel><PanelHeading title="Latest anomaly signals" description="Outliers include resource-level evidence and cost impact." action={<button className="text-button" onClick={() => go('anomalies')}>Open anomaly center <ChevronRight size={14} /></button>} /><AnomalyList anomalies={anomalies.slice(0, 3)} /></Panel></div>
    <div className="demo-note"><HelpCircle size={15} /><span>All data and rates are synthetic demo assumptions. No cloud resources are modified by FinOptiX.</span></div>
  </>;
}

function RecommendationTable({ recommendations, compact = false, onSelect }) {
  if (!recommendations.length) return <EmptyState title="No recommendations found" text="The current dataset has no resources meeting the configured optimization rules." />;
  return <div className="table-scroll"><table className="data-table"><thead><tr><th>RESOURCE</th><th>PROPOSED ACTION</th><th>SAVING / MONTH</th><th>RISK</th><th>CONFIDENCE</th><th /></tr></thead><tbody>{recommendations.map((row, index) => <tr key={row.option_id || resourceKey(row)} onClick={onSelect} className={onSelect ? 'clickable-row' : ''}><td><b>{row.resource_id}</b><small>{row.cloud_provider} · {row.service} · {row.evidence?.cpu_pct ?? '—'}% CPU</small></td><td><b className="action-copy">{row.action}</b><small>{row.reason}</small></td><td className="saving-cell">{money(row.estimated_monthly_saving)}<small>{compact ? `Priority ${row.priority_score}` : `${money(row.estimated_annual_saving)} / year`}</small></td><td><RiskTag risk={row.risk} /></td><td>{pct(Number(row.confidence) * 100, 0)}</td><td><ChevronRight size={15} className="muted-icon" /></td></tr>)}</tbody></table></div>;
}

function CostExplorer({ data }) {
  const [dimension, setDimension] = useState('by_service');
  const [query, setQuery] = useState('');
  const [uploadState, setUploadState] = useState('');
  const [uploadBusy, setUploadBusy] = useState(false);
  const dims = [{ id: 'by_service', label: 'Service' }, { id: 'by_provider', label: 'Provider' }, { id: 'by_region', label: 'Region' }, { id: 'by_department', label: 'Department' }, { id: 'by_project', label: 'Project' }];
  const values = data.summary[dimension] || {};
  const chartData = Object.entries(values).map(([name, cost]) => ({ name, cost: Number(cost) })).filter(x => x.name.toLowerCase().includes(query.toLowerCase()));
  const handleUpload = async (event) => {
    const file = event.target.files?.[0]; if (!file) return;
    setUploadBusy(true); setUploadState('');
    try { const form = new FormData(); form.append('file', file); const result = await request('/upload', { method: 'POST', body: form }); setUploadState(`Loaded ${result.loaded_rows.toLocaleString()} rows across ${result.resources} resources. Refresh the page data to see the new totals.`); }
    catch (err) { setUploadState(err.message); }
    finally { setUploadBusy(false); event.target.value = ''; }
  };
  const resources = data.summary.top_resources || [];
  return <>
    <PageHeading eyebrow="COST MANAGEMENT / EXPLORER" title="Explore your cloud spend" subtitle="Break down monthly resource cost by provider, service, geography, and ownership." action={<label className="button secondary upload-button"><UploadCloud size={15} />{uploadBusy ? 'Uploading…' : 'Upload CSV'}<input type="file" accept=".csv,text/csv" onChange={handleUpload} disabled={uploadBusy} /></label>} />
    {uploadState && <div className={uploadState.startsWith('Loaded') ? 'inline-success' : 'inline-warning'}>{uploadState.startsWith('Loaded') ? <Check size={15} /> : <AlertTriangle size={15} />}{uploadState}</div>}
    <div className="metric-grid three"><Metric label="MONTHLY RUN-RATE" value={money(data.summary.total_monthly_cost)} detail={`${data.summary.resource_count} unique resources`} icon={Cloud} /><Metric label="AVERAGE EFFICIENCY" value={pct(data.summary.average_efficiency, 1)} detail="Utilization-to-cost score" icon={Gauge} accent="blue" /><Metric label="COST DIMENSIONS" value={dims.length} detail="Provider, service, region & allocation" icon={Database} accent="violet" /></div>
    <Panel><div className="explorer-toolbar"><div><PanelHeading title="Cost allocation" description="Select a dimension to compare monthly run-rate." /></div><div className="toolbar-controls"><div className="search-field"><Search size={15} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Filter values" /></div><select value={dimension} onChange={e => setDimension(e.target.value)} aria-label="Cost dimension">{dims.map(item => <option value={item.id} key={item.id}>{item.label}</option>)}</select></div></div><div className="allocation-grid"><div className="allocation-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData} margin={{ top: 15, right: 15, left: 4, bottom: 8 }}><CartesianGrid stroke="#edf1ef" vertical={false} /><XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} angle={chartData.length > 6 ? -18 : 0} textAnchor={chartData.length > 6 ? 'end' : 'middle'} height={chartData.length > 6 ? 54 : 28} /><YAxis tickFormatter={compactMoney} tickLine={false} axisLine={false} width={58} /><Tooltip formatter={v => money(v)} /><Bar dataKey="cost" name="Monthly cost" fill="#4c9b81" radius={[5, 5, 0, 0]} barSize={34} /></BarChart></ResponsiveContainer></div><div className="allocation-list">{chartData.map((item, i) => <div className="allocation-row" key={item.name}><span className="rank-number">{String(i + 1).padStart(2, '0')}</span><span className="allocation-name">{item.name}</span><div className="allocation-meter"><span style={{ width: `${Math.max(4, (item.cost / Math.max(...chartData.map(x => x.cost), 1)) * 100)}%` }} /></div><b>{money(item.cost)}</b></div>)}</div></div></Panel>
    <Panel><PanelHeading title="Highest monthly cost resources" description="Latest observation for each provider/resource pair." /><div className="table-scroll"><table className="data-table"><thead><tr><th>RESOURCE</th><th>SERVICE</th><th>REGION</th><th>MONTHLY COST</th></tr></thead><tbody>{resources.map(row => <tr key={`${row.cloud_provider}:${row.resource_id}`}><td><b>{row.resource_id}</b><small>{row.cloud_provider}</small></td><td>{row.service}</td><td>{row.region}</td><td className="saving-cell">{money(row.monthly_cost)}</td></tr>)}</tbody></table></div></Panel>
  </>;
}

function ForecastPage({ forecast, trends }) {
  const series = forecast.series || [];
  const future = forecast.next_month;
  return <>
    <PageHeading eyebrow="PREDICTIVE ANALYTICS / FORECAST" title="Cloud cost forecast" subtitle="Time-ordered holdout results and forward cost estimates from the current demo model." />
    <div className="metric-grid three"><Metric label="NEXT DAY" value={money(forecast.next_day)} detail="Forecast cost" icon={Activity} /><Metric label="NEXT 7 DAYS" value={money(forecast.next_week)} detail="Cumulative forecast" icon={TrendingUp} accent="blue" /><Metric label="NEXT 30 DAYS" value={money(future)} detail={forecast.model || 'Forecast model'} icon={BarChart3} accent="violet" /></div>
    <Panel><PanelHeading title="Actual vs model prediction" description="Chronological test window. Earlier dates are actual-only; holdout dates include prediction." action={<span className="subtle-chip">{forecast.model || 'Baseline model'}</span>} /><div className="chart-area tall"><ResponsiveContainer width="100%" height="100%"><LineChart data={series} margin={{ top: 10, right: 18, bottom: 5, left: 4 }}><CartesianGrid stroke="#edf1ef" vertical={false} /><XAxis dataKey="date" tickFormatter={dateLabel} minTickGap={28} tickLine={false} axisLine={false} /><YAxis tickFormatter={compactMoney} width={60} tickLine={false} axisLine={false} /><Tooltip content={<ChartTooltip />} /><Legend /><Line type="monotone" dataKey="actual" name="Actual" stroke="#2d8b6b" strokeWidth={2} dot={false} connectNulls /><Line type="monotone" dataKey="predicted" name="Prediction" stroke="#6578cb" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls /></LineChart></ResponsiveContainer></div></Panel>
    <Panel><PanelHeading title="Candidate model comparison" description="The selected strategy is chosen on a chronological validation slice from training history. The later holdout is reserved for reporting." /><div className="table-scroll"><table className="data-table"><thead><tr><th>MODEL</th><th>VALIDATION MAE</th><th>HOLDOUT MAE</th><th>HOLDOUT RMSE</th><th>HOLDOUT MAPE</th><th>HOLDOUT R²</th></tr></thead><tbody>{(forecast.model_comparison || []).map(row => <tr key={row.model}><td><b>{row.model}{row.model === forecast.model ? ' · selected' : ''}</b></td><td>{money(row.validation_mae)}</td><td>{money(row.mae)}</td><td>{money(row.rmse)}</td><td>{pct(row.mape_pct, 2)}</td><td>{row.r2 ?? '—'}</td></tr>)}</tbody></table></div>{forecast.selection && <p className="muted">{forecast.selection.method}. Holdout R² can be negative when the forecast is worse than predicting the holdout mean; treat the model as a prototype where that occurs.</p>}</Panel>
    <div className="content-grid two-col"><Panel><PanelHeading title="Chronological holdout metrics" description="Compared with a seasonal-naive previous-week baseline." /><div className="metric-rows">{[['Model', forecast.model || 'Not reported'], ['Model MAE', money(forecast.metrics?.mae)], ['Model RMSE', money(forecast.metrics?.rmse)], ['Model MAPE', pct(forecast.metrics?.mape_pct, 2)], ['Model R²', forecast.metrics?.r2 ?? '—'], ['Naive baseline MAE', money(forecast.baseline_metrics?.mae)], ['Naive baseline MAPE', pct(forecast.baseline_metrics?.mape_pct,2)], ['Naive baseline R²', forecast.baseline_metrics?.r2 ?? '—']].map(([label, value]) => <div key={label}><span>{label}</span><b>{value}</b></div>)}</div></Panel><Panel className="method-note"><PanelHeading title="How to read this" /><p>Forecast values are estimates from the dataset currently supplied to the API. Holdout metrics summarize error on later dates; they are not a guarantee of future spend.</p><p>This is a single portfolio forecast. Provider and service forecasts are not separate model outputs. Negative R² indicates weak holdout fit; review the candidate comparison before using the projection.</p><div className="note-chip"><ShieldCheck size={15} /> Validate against real billing exports before financial planning.</div></Panel></div>
  </>;
}

function AnomalyList({ anomalies }) {
  if (!anomalies.length) return <EmptyState title="No anomalies in the current response" text="Refresh after the backend finishes analysis or check the uploaded dataset." />;
  return <div className="anomaly-list">{anomalies.map((item, i) => { const root = item.root_cause_analysis; return <div className="anomaly-item" key={`${item.cloud_provider}-${item.resource_id}-${item.timestamp}-${i}`}><div className={`anomaly-symbol ${item.severity?.toLowerCase()}`}><AlertTriangle size={16} /></div><div className="anomaly-main"><div className="anomaly-title"><b>{item.resource_id}</b><span>{item.cloud_provider} · {item.service} · {item.region}</span><RiskTag risk={item.severity} /></div><p>{item.cause}</p><small>{dateLabel(item.timestamp)} · impact {money(item.cost_impact_vs_baseline ?? item.cost_impact_vs_median)}</small>{root && <details className="root-cause-detail"><summary>Measured root-cause evidence</summary><p>{root.explanation}</p><small>Baseline {root.baseline_window?.join(' to ') || '—'} · recent {root.recent_window?.join(' to ') || '—'} · unit-price history {root.unit_price_change || 'unavailable'}</small>{root.resources_not_observed_in_baseline?.length>0 && <div className="root-cause-contribution"><b>Resource IDs seen recently but absent from baseline</b><span>{root.resources_not_observed_in_baseline.join(', ')}. {root.resource_presence_note}</span></div>}{root.service_region_contributions?.slice(0,3).map((part,index)=><div className="root-cause-contribution" key={index}><b>{part.cloud_provider} · {part.service} · {part.region}</b><span>{money(part.spend_delta)} spend change · resources {part.resource_count_before} → {part.resource_count_after} · runtime {part.runtime_hours_before}h → {part.runtime_hours_after}h · CPU {pct(part.cpu_mean_before,0)} → {pct(part.cpu_mean_after,0)} · memory {pct(part.memory_mean_before,0)} → {pct(part.memory_mean_after,0)}</span></div>)}</details>}</div></div>; })}</div>;
}
function AnomaliesPage({ anomalies }) {
  const [filter, setFilter] = useState('ALL');
  const visible = anomalies.filter(row => filter === 'ALL' || row.severity === filter);
  return <>
    <PageHeading eyebrow="DETECTION / ROOT-CAUSE ANALYSIS" title="Anomalies & root causes" subtitle="Inspect outlier signals with affected resources, services, regions, and supporting evidence." />
    <div className="metric-grid three"><Metric label="DETECTED SIGNALS" value={anomalies.length} detail="Isolation Forest candidates" icon={AlertTriangle} accent="amber" /><Metric label="HIGH SEVERITY" value={anomalies.filter(row => row.severity === 'HIGH').length} detail="Based on deviation from baseline" icon={Activity} accent="blue" /><Metric label="AFFECTED RESOURCES" value={new Set(anomalies.map(row => `${row.cloud_provider}:${row.resource_id}`)).size} detail="Provider-scoped resource IDs" icon={Database} accent="violet" /></div>
    <Panel><div className="filter-toolbar"><PanelHeading title="Anomaly timeline" description="Each explanation is derived from observed history." /><div className="segmented">{['ALL', 'HIGH', 'MEDIUM', 'LOW'].map(value => <button className={filter === value ? 'selected' : ''} onClick={() => setFilter(value)} key={value}>{value}</button>)}</div></div><AnomalyList anomalies={visible} /></Panel>
    <Panel className="method-note"><PanelHeading title="Detection method" /><p>The API applies Isolation Forest to changes in cost, CPU, memory, network, running time, and storage. Ground-truth synthetic labels are reserved for evaluation and are not detector inputs.</p><div className="note-chip"><HelpCircle size={15} /> Root-cause evidence shows associations in the dataset; it does not prove causation.</div></Panel>
  </>;
}

function RecommendationsPage({ recommendations, ranking, go }) {
  const [filter, setFilter] = useState('ALL');
  const [rankMode, setRankMode] = useState('risk_aware');
  const commonCandidates = ranking?.[rankMode] || recommendations;
  const filtered = commonCandidates.filter(row => filter === 'ALL' || row.risk === filter).map(row => ({...row, priority_score: rankMode === 'risk_aware' ? row.risk_aware_rank : row.savings_only_rank}));
  const total = recommendations.reduce((sum, row) => sum + Number(row.estimated_monthly_saving || 0), 0);
  return <>
    <PageHeading eyebrow="DECISION SUPPORT / OPTIMIZATION" title="Optimization recommendations" subtitle="Transparent rule-based opportunities ranked with confidence and operational risk." action={<button className="button primary" onClick={() => go('simulator')}><Play size={15} /> Run what-if simulation</button>} />
    <div className="metric-grid three"><Metric label="MONTHLY SAVING POTENTIAL" value={money(total)} detail="Sum of currently qualified opportunities" icon={ArrowDownRight} /><Metric label="LOW-RISK OPTIONS" value={recommendations.filter(row => row.risk === 'LOW').length} detail="Of the listed recommendations" icon={ShieldCheck} accent="blue" /><Metric label="AVERAGE CONFIDENCE" value={pct(recommendations.length ? recommendations.reduce((sum, row) => sum + Number(row.confidence || 0), 0) / recommendations.length * 100 : 0)} detail="Rule-evidence score, not probability" icon={Gauge} accent="violet" /></div>
    <Panel><div className="filter-toolbar"><PanelHeading title={`Ranked action/resource pairs (${filtered.length})`} description="Both methods rank the same feasible candidate pool." /><div className="rank-controls"><div className="segmented"><button className={rankMode === 'savings_only' ? 'selected' : ''} onClick={() => setRankMode('savings_only')}>Savings only</button><button className={rankMode === 'risk_aware' ? 'selected' : ''} onClick={() => setRankMode('risk_aware')}>Risk aware</button></div><div className="segmented">{['ALL', 'LOW', 'MEDIUM', 'HIGH'].map(value => <button className={filter === value ? 'selected' : ''} onClick={() => setFilter(value)} key={value}>{value}</button>)}</div></div></div><RecommendationTable recommendations={filtered} onSelect={() => go('simulator')} /></Panel>
    <Panel className="method-note"><PanelHeading title="Ranking objective" /><p>{ranking?.formula || 'Risk-aware ranking uses savings, evidence quality, and operational risk.'} Savings-only ranks gross monthly savings. Both methods use {ranking?.candidate_count || 0} identical feasible candidates. Evidence confidence is a data-quality score, not an optimization success probability.</p></Panel>
  </>;
}

function SimulatorPage({ recommendations }) {
  const [selectedKey, setSelectedKey] = useState(recommendations[0] ? resourceKey(recommendations[0]) : '');
  const selected = recommendations.find(row => resourceKey(row) === selectedKey) || recommendations[0];
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => { if (selected && !recommendations.some(row => resourceKey(row) === selectedKey)) setSelectedKey(resourceKey(selected)); }, [selected, selectedKey, recommendations]);
  const chooseResource = (value) => { setSelectedKey(value); setResult(null); setError(''); };
  const run = async () => {
    if (!selected) return;
    setBusy(true); setError('');
    try { setResult(await request('/what-if', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ resource_id: selected.resource_id, cloud_provider: selected.cloud_provider, action: 'compare' }) })); }
    catch (err) { setError(err.message); setResult(null); }
    finally { setBusy(false); }
  };
  return <>
    <PageHeading eyebrow="RISK-AWARE DECISION LAB / WHAT-IF" title="What-if simulator" subtitle="Preview an optimization’s cost, utilization, confidence, and risk before taking action." />
    <div className="simulator-layout"><Panel className="simulator-controls"><PanelHeading title="Configure a comparison" description="Choose a resource; all catalog-valid alternatives are evaluated." /><label className="field-label">RESOURCE</label><select value={selected ? resourceKey(selected) : ''} onChange={e => chooseResource(e.target.value)}>{recommendations.map(row => <option key={resourceKey(row)} value={resourceKey(row)}>{row.cloud_provider} · {row.resource_id} · {row.service}</option>)}</select>{selected && <div className="resource-evidence"><div><span>Observed monthly run-rate</span><b>{money(selected.evidence?.monthly_cost)}</b></div><div><span>Mean / p95 CPU</span><b>{pct(selected.evidence?.cpu_pct, 1)} / {pct(selected.evidence?.cpu_p95_pct, 1)}</b></div><div><span>Mean / p95 memory</span><b>{pct(selected.evidence?.memory_pct, 1)} / {pct(selected.evidence?.memory_p95_pct, 1)}</b></div><div><span>Historical coverage</span><b>{pct(Number(selected.evidence?.coverage_90d || 0) * 100, 0)}</b></div></div>}<button className="button primary full-width" onClick={run} disabled={busy || !selected}><Play size={15} />{busy ? 'Calculating alternatives…' : 'Compare feasible options'}</button>{error && <div className="inline-warning"><AlertTriangle size={15} />{error}</div>}<div className="sim-assumptions"><HelpCircle size={14} /><span>Rates are illustrative catalog assumptions in INR, not current provider quotes. No cloud resource is modified.</span></div></Panel>
      <div className="simulator-results">{result ? <><Panel className="result-panel"><div className="result-heading"><div><div className="eyebrow">OBSERVED BASELINE</div><h2>{result.cloud_provider} · {result.resource_id}</h2><p>{result.current?.observations} recent observations · {result.catalog_version}</p></div><span className="subtle-chip">{result.data_sufficient ? 'Evidence sufficient' : 'Insufficient evidence'}</span></div><div className="comparison"><div><span>CURRENT MONTHLY RUN-RATE</span><b>{money(result.current?.monthly_cost)}</b><small>{result.current?.instance_type} · observed mean</small></div><div className="comparison-arrow"><ChevronRight size={20} /></div><div><span>FEASIBLE ALTERNATIVES</span><b>{result.alternatives?.length || 0}</b><small>{result.pricing_source} pricing</small></div></div>{result.warning && <div className="inline-warning"><AlertTriangle size={15} />{result.warning}</div>}{result.alternatives?.length ? <div className="option-grid">{result.alternatives.map(option => <article className="option-card" key={option.option_id}><div className="option-heading"><b>{option.action.replaceAll('_',' ')}</b><RiskTag risk={option.risk} /></div><small>{option.target_instance || 'Scheduled action'} · risk score {option.risk_score}/100</small><div className="comparison option-comparison"><div><span>CURRENT</span><b>{money(option.current_monthly_cost)}</b></div><div className="comparison-arrow"><ChevronRight size={18}/></div><div><span>PROPOSED</span><b>{money(option.proposed_monthly_cost)}</b></div></div><div className="saving-highlight"><div><span>MONTHLY</span><b>{money(option.monthly_saving)}</b></div><div><span>ANNUALIZED</span><b>{money(option.annualized_saving)}</b></div></div><div className="post-metrics"><div><span>CPU mean / p95</span><b>{pct(option.expected_cpu_mean_pct,1)} / {pct(option.expected_cpu_p95_pct,1)}</b></div><div><span>Memory mean / p95</span><b>{pct(option.expected_memory_mean_pct,1)} / {pct(option.expected_memory_p95_pct,1)}</b></div><div><span>Evidence quality</span><b>{pct(Number(option.evidence_confidence)*100,0)}</b></div><div><span>Priority score</span><b>{option.risk_aware_score}</b></div></div><div className="option-decision">{option.decision_explanation}</div><details><summary>Assumptions and limitations</summary><ul className="assumption-list">{[...(option.assumptions||[]),...(option.limitations||[])].map((line,i)=><li key={i}>{line}</li>)}</ul></details></article>)}</div> : <EmptyState title="No feasible optimization found" text="The catalog, workload constraints, or insufficient history ruled out all proposed actions." />}</Panel>{result.rejected_options?.length>0 && <Panel><PanelHeading title={`Rejected options (${result.rejected_options.length})`} description="These alternatives were excluded because they violate constraints or lack required data." /><div className="rejected-list">{result.rejected_options.map((option,i)=><div key={`${option.option_id||option.action}-${i}`}><b>{option.action}{option.target_instance?` · ${option.target_instance}`:''}</b><span>{(option.performance_constraint_violations||[option.reason||'Insufficient evidence']).join('; ')}</span></div>)}</div></Panel>}</> : <Panel className="empty-simulation"><div className="empty-illustration"><Activity size={28} /></div><h2>Your scenario comparison will appear here</h2><p>Select a resource and evaluate every supported alternative against recent utilization history.</p><div className="empty-checks"><span><Check size={14} /> Catalog-based cost</span><span><Check size={14} /> Peak constraint checks</span><span><Check size={14} /> Evidence and risk scores</span></div></Panel>}</div></div>
  </>;
}

function SavingsPage({ recommendations, savings }) {
  const monthly = Number(savings.monthly_opportunity ?? recommendations.reduce((sum, row) => sum + Number(row.estimated_monthly_saving || 0), 0));
  const annual = Number(savings.annual_opportunity ?? monthly * 12);
  const byAction = Object.values(recommendations.reduce((groups, row) => {
    const name = row.action || 'Other';
    groups[name] ||= { name, monthly: 0, resources: 0 };
    groups[name].monthly += Number(row.estimated_monthly_saving || 0);
    groups[name].resources += 1;
    return groups;
  }, {})).sort((a, b) => b.monthly - a.monthly);
  const lowRisk = recommendations.filter(row => row.risk === 'LOW').reduce((sum, row) => sum + Number(row.estimated_monthly_saving || 0), 0);
  const highRisk = recommendations.filter(row => row.risk === 'HIGH').reduce((sum, row) => sum + Number(row.estimated_monthly_saving || 0), 0);
  return <>
    <PageHeading eyebrow="FINANCIAL IMPACT / SAVINGS" title="Savings analysis" subtitle="Review modeled savings by action and risk before prioritizing optimization work." action={<span className="subtle-chip">Synthetic INR estimates</span>} />
    <div className="metric-grid three"><Metric label="MONTHLY OPPORTUNITY" value={money(monthly)} detail="Sum of rule-based estimates" icon={CircleDollarSign} /><Metric label="ANNUALIZED OPPORTUNITY" value={money(annual)} detail="Monthly estimate × 12" icon={TrendingUp} accent="blue" /><Metric label="LOW-RISK SAVING SHARE" value={money(lowRisk)} detail={`${recommendations.filter(row => row.risk === 'LOW').length} low-risk recommendations`} icon={ShieldCheck} accent="violet" /></div>
    <div className="content-grid two-col"><Panel><PanelHeading title="Potential savings by action" description="Monthly opportunity estimated from the recommendation rules." /><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={byAction} margin={{ top: 14, right: 10, left: 2, bottom: 8 }}><CartesianGrid stroke="#edf1ef" vertical={false} /><XAxis dataKey="name" tickLine={false} axisLine={false} interval={0} /><YAxis tickFormatter={compactMoney} tickLine={false} axisLine={false} width={55} /><Tooltip formatter={value => money(value)} /><Bar dataKey="monthly" name="Monthly savings" fill="#4d9d7f" radius={[5, 5, 0, 0]} barSize={38} /></BarChart></ResponsiveContainer></div></Panel><Panel><PanelHeading title="Risk-adjusted savings mix" description="Gross savings shown alongside operational risk." /><div className="savings-mix">{['LOW', 'MEDIUM', 'HIGH'].map(risk => { const rows = recommendations.filter(row => row.risk === risk); const value = rows.reduce((sum, row) => sum + Number(row.estimated_monthly_saving || 0), 0); return <div className="savings-risk-row" key={risk}><div><RiskTag risk={risk} /><span>{rows.length} opportunities</span></div><b>{money(value)}<small>/ month</small></b><div className="savings-track"><span className={`track-${risk.toLowerCase()}`} style={{ width: `${monthly ? value / monthly * 100 : 0}%` }} /></div></div>; })}<div className="note-chip"><HelpCircle size={15} /> Estimates use demo heuristics. Simulate an action and validate service objectives before implementation.</div>{highRisk > 0 && <div className="inline-warning"><AlertTriangle size={15} />{money(highRisk)} per month is tied to high-risk options; review capacity headroom before prioritizing.</div>}</div></Panel></div>
    <Panel><PanelHeading title="Savings opportunities" description="Select an opportunity in Recommendations to open it in the simulator." /><RecommendationTable recommendations={recommendations} /></Panel>
  </>;
}

function CopilotPage() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const prompts = ['Why did our cloud cost increase?', 'What are the top savings opportunities?', 'Which recommendation has the lowest risk?', 'Explain the latest anomaly.'];
  const submit = async (value = question) => {
    const q = value.trim(); if (q.length < 3 || busy) return;
    setBusy(true); setError(''); setQuestion(''); setMessages(prev => [...prev, { role: 'user', text: q }]);
    try { const reply = await request('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) }); setMessages(prev => [...prev, { role: 'assistant', text: reply.answer, sources: reply.sources, grounding: reply.grounding, mode: reply.mode }]); }
    catch (err) { setError(err.message); }
    finally { setBusy(false); }
  };
  return <>
    <PageHeading eyebrow="ASSISTANT / PROJECT ANALYTICS" title="FinOps copilot" subtitle="Ask questions about the currently loaded synthetic usage data and recommendations." />
    <div className="copilot-layout"><Panel className="chat-panel"><div className="chat-intro"><div className="copilot-orb"><Bot size={23} /></div><div><b>FinOptiX copilot</b><span>Structured application analytics and local knowledge retrieval.</span></div><span className="grounded-chip"><span /> GROUNDED</span></div><div className="chat-body">{messages.length === 0 ? <div className="chat-empty"><div className="chat-spark"><Sparkles size={19} /></div><h2>What would you like to know?</h2><p>Ask about live spend, anomalies, risk, or savings—or ask for general cloud optimization guidance with source references.</p><div className="prompt-grid">{prompts.map(prompt => <button onClick={() => submit(prompt)} key={prompt}>{prompt}<ChevronRight size={14} /></button>)}</div></div> : <div className="messages">{messages.map((message, index) => <div className={`message ${message.role}`} key={`${index}-${message.text}`}><div className="message-avatar">{message.role === 'assistant' ? <Bot size={15} /> : 'You'}</div><div className="message-content"><b>{message.role === 'assistant' ? `FinOptiX · ${message.mode === 'knowledge_retrieval' ? 'Knowledge' : 'Analytics'}` : 'You'}</b><p>{message.text}</p>{message.role === 'assistant' && <small>{message.grounding || 'Source: live analytics dataset'}{message.sources?.length ? <span className="chat-sources">{message.sources.map((source, i) => <a key={i} href={typeof source === 'string' ? undefined : source.reference} target="_blank" rel="noreferrer">{typeof source === 'string' ? source : source.title}</a>)}</span> : ''}</small>}</div></div>)}</div>}{busy && <div className="typing"><span /><span /><span /> Checking project data and indexed guidance…</div>}{error && <div className="inline-warning"><AlertTriangle size={15} />{error}</div>}</div><form className="chat-composer" onSubmit={e => { e.preventDefault(); submit(); }}><input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask about spend, risk, or cloud optimization guidance…" /><button className="button primary" disabled={busy || question.trim().length < 3}><MessageCircle size={15} /> Ask</button></form></Panel><Panel className="copilot-side"><div className="assistant-visual"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="assistant-core"><Sparkles size={25} /></div></div><h2>Two grounded answer modes</h2><p>Analytics answers use validated FinOptiX calculations. Knowledge answers retrieve passages from indexed local documentation and show the source link.</p><div className="trust-row"><ShieldCheck size={16} /><span>Financial values come from application analytics</span></div><div className="trust-row"><Database size={16} /><span>Knowledge base has source-attributed documents</span></div><div className="trust-row"><HelpCircle size={16} /><span>External generation is optional; retrieval works without a key</span></div></Panel></div>
  </>;
}

function EmptyState({ title, text }) { return <div className="empty-state"><div><Database size={18} /></div><b>{title}</b><p>{text}</p></div>; }

createRoot(document.getElementById('root')).render(<App />);
