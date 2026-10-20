/** API client — typed against ../openapi.json. No business logic here. */
export type Estimator = 'linear_dml' | 'causal_forest' | 's_learner' | 't_learner' | 'x_learner';
export interface RunConfig { seed: number; estimator: Estimator; n_folds: number; budget: number; unit_cost: number; margin: number; source: 'simulator' | 'criteo'; n_rows: number; target: 'visit' | 'conversion'; }
export const DEFAULT_CONFIG: RunConfig = { seed: 42, estimator: 'causal_forest', n_folds: 5, budget: 1000, unit_cost: 0.1, margin: 20, source: 'simulator', n_rows: 20000, target: 'visit' };
async function j<T>(r: Response): Promise<T> { if (!r.ok) throw new Error(`${r.status} ${await r.text()}`); return r.json() as Promise<T>; }
export const api = {
  createRun: (cfg: RunConfig) => fetch('/v1/runs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...cfg, nuisance: 'lightgbm', trim: [0.025, 0.975] }) }).then(j<{ run_id: string } >),
  fit: (id: string) => fetch(`/v1/runs/${id}/fit`, { method: 'POST' }).then(j<{ ate_hat: number; eval_n: number; overlap_gate: Record<string, unknown> } >),
  effects: (id: string) => fetch(`/v1/runs/${id}/effects`).then(j<{ tau_hat: number[]; lo: number[] | null; hi: number[] | null } >),
  policy: (id: string) => fetch(`/v1/runs/${id}/policy:optimize`, { method: 'POST' }).then(j<{ assignment: number[]; expected_profit: number; spent: number; method: string } >),
  report: (id: string) => fetch(`/v1/runs/${id}/report`).then(j<Record<string, unknown>>),
  calibration: (id: string) => fetch(`/v1/runs/${id}/calibration`).then(j<{ deciles: Record<string, number>[] } >),
  graph: (source: string) => fetch('/v1/graph/identify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ source }) }).then(j<{ graph: string; backdoor_vars: string[] } >),
  nl: (query: string) => fetch('/v1/nl/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }) }).then(j<{ run_config: RunConfig; valid: boolean } >),
};
