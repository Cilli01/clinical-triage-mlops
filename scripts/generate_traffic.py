"""Gera trafego sintetico contra a API para validar o dashboard do Grafana.

Sem trafego, os paineis de RPS, latencia e distribuicao de predicoes ficam
vazios. Este script dispara chamadas variadas a /predict (casos normais, de
atencao e urgentes) e a /health, permitindo observar os paineis dinamicos
com dados reais.

Uso:
    uv run python scripts/generate_traffic.py --requests 200 --interval 0.2
"""

import argparse
import json
import time
import urllib.error
import urllib.request

ABSTRACTS_BY_URGENCY = {
    "urgent": [
        "Patient presenting acute cardiac symptoms and chest pain",
        "Suspected stroke with sudden brain infarction signs",
        "Severe sleep apnea episode with respiratory distress",
    ],
    "attention": [
        "Gastric infection with recurring stomach pain",
        "Suspicious hepatic tumor found on imaging",
        "Localized cancer under investigation, stable condition",
    ],
    "normal": [
        "Routine checkup, patient reports no significant symptoms",
        "Follow-up visit for a fully healed minor injury",
        "General wellness consultation, no abnormal findings",
    ],
}


def call_predict(base_url: str, urgency: str, abstract: str) -> None:
    payload = json.dumps({"condition_label": 1, "medical_abstract": abstract}).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        f"{base_url}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            print(f"[predict:{urgency}] status={response.status}")
    except urllib.error.URLError as error:
        print(f"[predict:{urgency}] falhou: {error}")


def call_health(base_url: str) -> None:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            print(f"[health] status={response.status}")
    except urllib.error.URLError as error:
        print(f"[health] falhou: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.3)
    args = parser.parse_args()

    urgencies = list(ABSTRACTS_BY_URGENCY.items())

    for i in range(args.requests):
        urgency, abstracts = urgencies[i % len(urgencies)]
        abstract = abstracts[i % len(abstracts)]

        call_predict(args.base_url, urgency, abstract)

        if i % 5 == 0:
            call_health(args.base_url)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
