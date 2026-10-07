## Table VI — Ablation Study (A2, fnr=0, L0)

| Condition | Defense proxy | N | ASR | 95% CI | Isolation | TTFT p50 | Components |
|---|---|---|---|---|---|---|---|
| AB0 None | D0_no_defense | 18 | 0.556 | (0.337,0.754) | 0.000 | 222.0 | none |
| AB1 Content | D3_binary_pii | 18 | 0.000 | (0.000,0.176) | 1.000 | 1784.8 | content_risk |
| AB2 Cont+Conf | AB2_content_confidence | 18 | 0.000 | (0.000,0.176) | 1.000 | 1785.5 | content+confidence |
| AB3 Behavior | AB3_behavior_only | 18 | 0.000 | (0.000,0.176) | 0.083 | 203.0 | behavior |
| FULL D5 | D5_risk_behavior | 18 | 0.000 | (0.000,0.176) | 1.000 | 1778.5 | content+confidence+behavior |
