#!/usr/bin/env python3
"""Terminal demonstration for OCG (layer) visibility queries.

Run from the repository root:

    python3 tools/layers_demo.py

The script builds the optional-content fixtures from
``tests/sourcemap_fixtures.py`` inside a temporary directory and shows:

1. A minimum-flip query over OCGs and OCMD memberships (And/Or/Not VE).
2. The concrete native extraction reproducing the target under the returned
   state, with hidden text excluded.
3. The 16-group / 4096-character fixture proving a global minimum rather than
   a first-feasible configuration.
4. The ValueError cases (missing group, cyclic VE, unbalanced marking).
"""

from __future__ import annotations

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdfminer.high_level import (  # noqa: E402
    extract_layered_text,
    find_layer_state_for_text,
)
from tests.sourcemap_fixtures import (  # noqa: E402
    write_cyclic_ve_variant,
    write_ocg_variant,
    write_sixteen_ocg_variant,
    write_unbalanced_marked_content_variant,
)


def heading(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        heading("1. Minimum-flip query over OCGs and And/Or/Not OCMD VE")
        ocg_path = tmp_dir / "ocg.pdf"
        write_ocg_variant(ocg_path)
        default_states: Dict[int, bool] = {10: False, 11: False}
        print("default state table:", default_states)
        for target in (
            "Always",
            "AlphaOne",
            "BothOnly",
            "EitherOne",
            "NotAlpha",
            "FormHidden",
            "MissingText",
        ):
            result = find_layer_state_for_text(
                ocg_path, target, default_states
            )
            if result is None:
                print(f"  {target:<11} -> None (no assignment reveals it)")
            else:
                print(
                    f"  {target:<11} -> {result.flips} flip(s), "
                    f"page {result.page_number}, states {result.states}"
                )

        heading("2. Returned state drives a concrete layered extraction")
        result = find_layer_state_for_text(
            ocg_path, "BothOnly", default_states
        )
        assert result is not None
        layered = extract_layered_text(ocg_path, result.states)
        print("states:", result.states)
        print("native visible sequence:", layered.page_text(0))
        print("target reproduced:", "BothOnly" in layered.page_text(0))

        heading("3. Sixteen groups / 4096 native characters (global optimum)")
        big_path = tmp_dir / "big.pdf"
        write_sixteen_ocg_variant(big_path)
        groups = list(range(20, 36))
        off_states = {objid: False for objid in groups}
        all_on = extract_layered_text(
            big_path, {objid: True for objid in groups}
        )
        total = sum(len(page) for page in all_on.pages)
        print("native characters when every layer is on:", total)
        cheap = find_layer_state_for_text(
            big_path, "EarlyFlipMarkerOneTwo34", off_states
        )
        expensive = find_layer_state_for_text(
            big_path, "NeedAll16", off_states
        )
        assert cheap is not None and expensive is not None
        print("one-group decoy cost :", cheap.flips, "flip(s)")
        print("conjunctive target   :", expensive.flips, "flips (global min)")

        heading("4. Invalid configurations raise ValueError, never a result")
        cyclic_path = tmp_dir / "cyclic.pdf"
        write_cyclic_ve_variant(cyclic_path)
        unbalanced_path = tmp_dir / "unbalanced.pdf"
        write_unbalanced_marked_content_variant(unbalanced_path)
        for label, path, states in (
            ("missing group", ocg_path, {10: False}),
            ("cyclic VE", cyclic_path, {10: False}),
            ("unbalanced BDC/EMC", unbalanced_path, {10: False}),
        ):
            try:
                find_layer_state_for_text(path, "X", states)
            except ValueError as exc:
                print(f"  {label:<18} -> ValueError: {exc}")
            else:  # pragma: no cover - defensive
                raise AssertionError(f"{label} unexpectedly succeeded")

        print("\nAll demonstrations completed.")


if __name__ == "__main__":
    main()
