"""Emit a small JSON view of a ROOT tree for the R9 integration tests."""

from __future__ import annotations

import json
import math
import sys

import ROOT

path, tree_name = sys.argv[1:3]
handle = ROOT.TFile.Open(path, "READ")
if not handle or handle.IsZombie():
    raise SystemExit(f"cannot open {path}")
tree = handle.Get(tree_name)
if not tree:
    raise SystemExit(f"tree {tree_name!r} not found in {path}")

branches = [
    {
        "name": branch.GetName(),
        "title": branch.GetTitle(),
        "class": branch.GetClassName(),
        "entries": int(branch.GetEntries()),
    }
    for branch in tree.GetListOfBranches()
]
scalar_names = [branch["name"] for branch in branches if not branch["class"]]
rows = []
for index in range(tree.GetEntries()):
    tree.GetEntry(index)
    row = {}
    for name in dict.fromkeys(scalar_names):
        value = getattr(tree, name)
        value = float(value) if isinstance(value, float) else int(value)
        row[name] = None if isinstance(value, float) and math.isnan(value) else value
    rows.append(row)

print(json.dumps({"entries": len(rows), "branches": branches, "rows": rows}, sort_keys=True))
handle.Close()
