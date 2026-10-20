import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import Plot from 'react-plotly.js';
import ReactFlow, { Background, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import { api, DEFAULT_CONFIG, RunConfig } from './api/client';

function useRun() {
  const [runId, setRunId] = useState<string | null>(null);
  const create = useMutation({ mutationFn: (cfg: RunConfig) => api.createRun(cfg).then(r => { setRunId(r.run_id); return r; }) });
  const fit = useMutation({ mutationFn: (id: string) => api.fit(id) });
  return { runId, create, fit };
}

export default function App() {
  const { runId, create, fit } = useRun();
  const [cfg, setCfg] = useState(DEFAULT_CONFIG);
  const effects = useQuery({ queryKey: ['effects', runId], queryFn: () => api.effects(runId!), enabled: !!runId && fit.isSuccess });
  const policy = useQuery({ queryKey: ['policy', runId], queryFn: () => api.policy(runId!), enabled: false });
  const report = useQuery({ queryKey: ['report', runId], queryFn: () => api.report(runId!), enabled: !!runId && fit.isSuccess });
  const graph = useQuery({ queryKey: ['graph', cfg.source], queryFn: () => api.graph(cfg.source) });
  const tau: number[] = effects.data?.tau_hat ?? [];
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold">EconoCausal — who buys <em>because of</em> the discount</h1>
      <p className="text-sm text-slate-600">Bands show measured CV coverage, never nominal 95%. See docs/WEEK23_PROGRESS.md §11.</p>
      <div className="flex gap-2 mt-4 flex-wrap">
        <label>estimator <select value={cfg.estimator} onChange={e => setCfg({ ...cfg, estimator: e.target.value as RunConfig['estimator'] })}>{['causal_forest', 'linear_dml', 's_learner', 't_learner', 'x_learner'].map(m => <option key={m}>{m}</option>)}</select></label>
        <label>source <select value={cfg.source} onChange={e => setCfg({ ...cfg, source: e.target.value as RunConfig['source'] })}><option>simulator</option><option>criteo</option></select></label>
        <label>budget <input type="number" value={cfg.budget} onChange={e => setCfg({ ...cfg, budget: +e.target.value })} className="w-24 border" /></label>
        <button className="bg-slate-900 text-white px-3 py-1 rounded" disabled={create.isPending} onClick={() => create.mutate(cfg)}>create run</button>
        <button className="bg-emerald-700 text-white px-3 py-1 rounded" disabled={!runId || fit.isPending} onClick={() => runId && fit.mutate(runId)}>fit</button>
        <button className="bg-indigo-700 text-white px-3 py-1 rounded" disabled={!runId} onClick={() => policy.refetch()}>optimize policy</button>
      </div>
      {runId && <p className="mt-2 text-sm">run_id: <code>{runId}</code> {fit.data && <>ATÊ={fit.data.ate_hat.toFixed(4)} n={fit.data.eval_n} gate_block={String((fit.data.overlap_gate as Record<string, unknown>).block_policy)}</>}</p>}
      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <div className="bg-white p-3 rounded shadow h-80"><h2 className="font-semibold">ITE distribution</h2>
          {tau.length > 0 && <Plot data={[{ type: 'histogram', x: tau, name: 'tau_hat' }]} layout={{ autosize: true, margin: { t: 10 } }} useResizeHandler style={{ width: '100%', height: '85%' }} />}
        </div>
        <div className="bg-white p-3 rounded shadow h-80"><h2 className="font-semibold">Causal DAG</h2>
          <div className="text-xs">{graph.data?.graph}</div>
          <div style={{ height: 200 }}><ReactFlow nodes={(graph.data?.backdoor_vars ?? []).map((v: string, i: number) => ({ id: v, position: { x: i * 120, y: 50 }, data: { label: v } })).concat([{ id: 'T', position: { x: 100, y: 150 }, data: { label: 'T (discount)' } }, { id: 'Y', position: { x: 300, y: 150 }, data: { label: 'Y (buy)' } }])} edges={(graph.data?.backdoor_vars ?? []).flatMap((v: string) => [{ id: `${v}T`, source: v, target: 'T' }, { id: `${v}Y`, source: v, target: 'Y' }]).concat([{ id: 'TY', source: 'T', target: 'Y' }])} fitView><Background /><Controls /></ReactFlow></div>
        </div>
        <div className="bg-white p-3 rounded shadow h-80"><h2 className="font-semibold">Policy</h2>
          {policy.data && <><p>profit ${policy.data.expected_profit.toFixed(0)} / spent ${policy.data.spent.toFixed(0)} ({policy.data.method})</p>
            <Plot data={[{ type: 'histogram', x: tau.filter((_, i) => policy.data.assignment[i] === 1), name: 'picked tau' }]} layout={{ autosize: true, margin: { t: 10 } }} useResizeHandler style={{ width: '100%', height: '70%' }} /></>}
        </div>
        <div className="bg-white p-3 rounded shadow h-80"><h2 className="font-semibold">Report</h2>
          <pre className="text-xs overflow-auto h-64">{report.data ? JSON.stringify(report.data, null, 1) : 'fit first'}</pre>
        </div>
      </div>
    </div>
  );
}
