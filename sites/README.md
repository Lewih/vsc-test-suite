# sites/

Each file in this directory defines the ReFrame system configuration for one
VSC site. The main config (`config_vsc.py`) auto-discovers these files at load
time — **adding a new site only requires dropping a new file here**.

## File naming

Files are named after the **city** of the university operating the cluster.

| File | University | Clusters |
|---|---|---|
| `brussel.py` | VUB | Hydra |
| `gent.py` | UGent | Hortense |
| `leuven.py` | KU Leuven | Genius |
| `antwerp.py` | UA (calcua) | Vaughan, Leibniz, Breniac |

## Contract

Each file must export a `systems` list of ReFrame system dicts:

```python
systems = [
    {'name': 'mycluster', 'hostnames': [...], 'partitions': [...], ...},
]
```

Single-cluster sites may also export a bare `system` dict — both forms are
recognised by the auto-discovery loop.

## cpu_env_list

`cpu_env_list` is defined in `config_vsc.py` and injected as a virtual
`sites.common` module before site files are loaded, so site files can do
`from sites.common import cpu_env_list` without a physical file and without
circular imports. To add or remove a toolchain, edit the list in `config_vsc.py`.
