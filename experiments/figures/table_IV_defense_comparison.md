## Table IV — Defense Comparison (A2, fnr=0, L0)

| Defense | Description | N | Residual ASR | 95% CI | Sensitive isolation | TTFT p50 | TTFT p95 | Tag |
|---|---|---|---|---|---|---|---|---|
| D0_no_defense | No defense (shared cache) | 18 | 0.556 | (0.337,0.754) | 0.000 | 222.0 | 589.3 | empirical |
| D1_full_isolation | Full isolation (cache disabled) | 18 | 0.000 | (0.000,0.176) | 1.000 | 1789.7 | 2228.7 | empirical |
| D2_tenant_salt | Tenant/user salting | 18 | 0.000 | (0.000,0.176) | 1.000 | 224.9 | 1071.2 | empirical |
| D3_binary_pii | Binary PII isolation | 18 | 0.000 | (0.000,0.176) | 1.000 | 1784.8 | 2220.2 | empirical |
| D4_risk_adaptive | Risk-adaptive (Issue #7) | 18 | 0.000 | (0.000,0.176) | 1.000 | 1786.4 | 2226.1 | empirical |
| D5_risk_behavior | Risk + behavior adaptive (Issue #10 proposed) | 18 | 0.000 | (0.000,0.176) | 1.000 | 1778.5 | 2213.7 | empirical |
| AB2_content_confidence | Ablation: content + confidence only | 18 | 0.000 | (0.000,0.176) | 1.000 | 1785.5 | 2219.3 | empirical |
| AB3_behavior_only | Ablation: behavior detection only | 18 | 0.000 | (0.000,0.176) | 0.083 | 203.0 | 1053.7 | empirical |
