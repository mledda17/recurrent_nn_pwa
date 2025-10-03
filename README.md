# RNN to PWA Equivalence

ReLU-RNN → PWA equivalence: scoperta regioni (pattern di attivazione), dinamiche locali
\((A_\sigma, B_\sigma, c_\sigma)\), simulazione RNN vs PWA, grafo strutturale di adiacenza tra regioni e
visualizzazioni “stile paper”.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
# oppure come package:
pip install -e .
# opzionale: supporto plot grafo
pip install -e .[graph]
