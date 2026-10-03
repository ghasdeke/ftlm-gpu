# Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
# and Helmholtz-Zentrum Dresden-Rossendorf e.V.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command line: ``python -m ftlm_gpu input.toml [-o results.npz|.mat]``."""

import argparse
from pathlib import Path

from .ftlm import run
from .io import load_input, save_results
from .model import Model


def main(argv=None):
    p = argparse.ArgumentParser(prog="ftlm_gpu",
                                description="Sector-FTLM thermodynamics of spin clusters")
    p.add_argument("input", help="input file (.toml or .json)")
    p.add_argument("-o", "--output", help="result file (.npz or .mat); default: output_dir/"
                   "output_name from the input file, else ftlm_<tag>.npz in the input file's "
                   "directory")
    p.add_argument("--backend", choices=["gpu", "cpu"])
    p.add_argument("--precision", choices=["single", "double", "half", "bfloat16"])
    p.add_argument("--lookup", choices=["clt", "cr"])
    args = p.parse_args(argv)

    opts = load_input(args.input)
    for key in ("backend", "precision", "lookup"):
        if getattr(args, key):
            opts[key] = getattr(args, key)
    model = Model.from_options(opts)
    res = run(opts, model=model)
    if args.output:
        out = Path(args.output)
    else:
        name = opts.get("output_name") or f"ftlm_{model.tag}.npz"
        out = Path(args.input).parent / opts.get("output_dir", ".") / name
        out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nResults saved to: {save_results(res, out)}")


if __name__ == "__main__":
    main()
