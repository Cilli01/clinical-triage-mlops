# Observabilidade e Monitoramento

Este documento detalha a instrumentação de métricas da API, os painéis do
dashboard Grafana e como validar localmente que a stack de observabilidade
está funcionando de ponta a ponta.

## 1. Métricas expostas em `/metrics`

A API expõe métricas no formato Prometheus através do endpoint `GET
/metrics`, coletadas por um middleware HTTP em `src/api/main.py` que envolve
todas as rotas.

| Métrica | Tipo | Labels | O que mede |
| :--- | :--- | :--- | :--- |
| `api_requests_total` | Counter | `endpoint`, `method`, `http_status` | Total de requisições recebidas, permitindo calcular RPS e volume por rota/status. |
| `api_request_latency_seconds` | Histogram | `endpoint` | Distribuição do tempo de resposta por rota, usada para calcular percentis (P95/P99). |
| `predictions_total` | Counter | `urgency_class` | Total de predições feitas pelo modelo, por classe de urgência (`urgent`, `attention`, `normal`). |
| `api_errors_total` | Counter | `endpoint`, `error_type` | Total de erros tratados na API (ex.: abstract vazio, exceções não tratadas). |

Essas métricas são cobertas por testes automatizados em `tests/test_metrics.py`,
que validam tanto o formato exposto quanto o incremento correto de cada
contador.

## 2. Painéis do dashboard Grafana

O dashboard `grafana/provisioning/dashboards/triagem-dashboard.json` é
provisionado automaticamente ao subir o Grafana (sem necessidade de
configuração manual) e contém 6 painéis:

1. **Requisições por segundo (RPS) — por endpoint**: `rate(api_requests_total[1m])` agrupado por `endpoint`. Mostra o volume de tráfego em tempo real.
2. **Latência P95 e P99 — por endpoint**: `histogram_quantile` sobre `api_request_latency_seconds_bucket`. Mostra se a API está respondendo dentro de um tempo aceitável, mesmo nos piores casos.
3. **Distribuição de predições por classe de urgência**: pizza com `sum(predictions_total) by (urgency_class)`. Mostra a proporção de laudos classificados como urgente/atenção/normal.
4. **Taxa de erros (%) — últimos 5 min**: proporção de respostas 4xx/5xx sobre o total de requisições. Serve de alerta visual rápido de instabilidade.
5. **Requisições totais (acumulado)**: contador simples do volume total de tráfego já servido.
6. **Erros por tipo (últimos 1 min)**: `rate(api_errors_total[1m])` agrupado por `error_type`, útil para diagnosticar qual tipo de erro está ocorrendo.

## 3. Como validar a stack localmente

```bash
# 1. Subir a stack completa (API + Prometheus + Grafana)
docker compose up --build -d

# 2. Confirmar que os 3 serviços estão "healthy"
docker compose ps

# 3. Gerar trafego sintetico para popular os paineis
uv run python scripts/generate_traffic.py --requests 200 --interval 0.2

# 4. Abrir o dashboard
# http://localhost:3000  (usuario: admin / senha: admin)
```

Uma evidência de execução real (screenshot do dashboard com dados) fica em
[`docs/evidencias/grafana-dashboard.png`](evidencias/grafana-dashboard.png).
