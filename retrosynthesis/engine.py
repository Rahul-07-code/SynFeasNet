from pathlib import Path

from rdkit import Chem

from .templates import REACTIONS
from .routes import (
    RouteNode,
    RetrosynthesisRoute,
    route_to_dict
)

from .utils import (
    validate_smiles,
    canonicalize,
    is_macrocycle
)

from .scorer import score_route


MAX_DEPTH = 4
BEAM_WIDTH = 3
MAX_ROUTES = 10


class RetrosynthesisEngine:

    def __init__(self):

        self.reactions = REACTIONS

        self.building_blocks = set()

        bb_file = (
            Path(__file__).parent /
            "building_blocks.txt"
        )

        if bb_file.exists():

            with open(bb_file, "r") as f:

                for line in f:

                    smiles = line.strip()

                    if smiles:
                        self.building_blocks.add(
                            canonicalize(smiles) or smiles
                        )

    def _is_building_block(
        self,
        smiles
    ):

        canonical = canonicalize(smiles)

        return (
            smiles in self.building_blocks
            or canonical in self.building_blocks
        )

    def _complexity(
        self,
        smiles
    ):

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return 999

        return (
            mol.GetNumAtoms()
            + mol.GetRingInfo().NumRings()
        )

    def _expand_tree(
        self,
        smiles,
        depth,
        visited,
        reaction=None
    ):

        node = RouteNode(
            smiles=smiles,
            depth=depth,
            reaction=reaction,
            is_building_block=self._is_building_block(smiles)
        )

        if depth >= MAX_DEPTH:
            return node

        if smiles in visited:
            return node

        if node.is_building_block:
            return node

        visited = visited.copy()
        visited.add(smiles)

        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return node

        generated_precursors = []
        seen_precursors = set()

        for rxn_name, rxn in self.reactions.items():

            try:

                outcomes = rxn.RunReactants(
                    (mol,)
                )

                outcomes = outcomes[:BEAM_WIDTH]

                for outcome in outcomes:

                    for precursor in outcome:

                        precursor_smiles = (
                            Chem.MolToSmiles(
                                precursor
                            )
                        )

                        precursor_smiles = (
                            canonicalize(precursor_smiles)
                            or precursor_smiles
                        )

                        if precursor_smiles in seen_precursors:
                            continue

                        seen_precursors.add(
                            precursor_smiles
                        )

                        generated_precursors.append(
                            (
                                rxn_name,
                                precursor_smiles
                            )
                        )

            except Exception:
                continue

        generated_precursors.sort(
            key=lambda item: self._complexity(item[1])
        )

        generated_precursors = (
            generated_precursors[:BEAM_WIDTH]
        )

        for rxn_name, precursor in generated_precursors:

            child = self._expand_tree(
                precursor,
                depth + 1,
                visited,
                reaction=rxn_name
            )

            node.children.append(
                child
            )

        return node

    def _count_nodes(
        self,
        node
    ):

        total = 1

        for child in node.children:

            total += self._count_nodes(
                child
            )

        return total

    def _count_solved(
        self,
        node
    ):

        solved = 0

        if (
            node.is_building_block
            or self._is_building_block(node.smiles)
        ):
            solved += 1

        for child in node.children:

            solved += self._count_solved(
                child
            )

        return solved

    def _route_depth(
        self,
        node
    ):

        if not node.children:
            return node.depth

        return max(
            self._route_depth(child)
            for child in node.children
        )

    def run(
        self,
        smiles,
        spi_score=0.8,
        sa_score=3.0,
        scscore=2.5,
        syba_score=10.0
    ):

        routes = []

        if not validate_smiles(
                smiles
        ):
            return routes

        if is_macrocycle(
                smiles
        ):
            return routes

        root = self._expand_tree(
            smiles,
            0,
            set()
        )

        total_nodes = self._count_nodes(
            root
        )

        solved_nodes = self._count_solved(
            root
        )

        solved_fraction = (
            solved_nodes /
            max(total_nodes, 1)
        )

        n_steps = self._route_depth(
            root
        )

        route_score = score_route(
            spi_score=spi_score,
            sa_score=sa_score,
            scscore=scscore,
            syba_score=syba_score,
            solved_fraction=solved_fraction,
            n_steps=n_steps
        )

        route = RetrosynthesisRoute(
            root=root,
            score=route_score,
            solved_fraction=solved_fraction,
            n_steps=n_steps
        )

        routes.append(route)

        routes.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return routes[:MAX_ROUTES]

    def run_json(
        self,
        smiles,
        spi_score=0.8,
        sa_score=3.0,
        scscore=2.5,
        syba_score=0.5
    ):

        routes = self.run(
            smiles=smiles,
            spi_score=spi_score,
            sa_score=sa_score,
            scscore=scscore,
            syba_score=syba_score
        )

        route_dicts = [
            route_to_dict(route, rank=index + 1)
            for index, route in enumerate(routes)
        ]

        best_route = route_dicts[0] if route_dicts else None

        return {
            "target_smiles": canonicalize(smiles) or smiles,
            "status": "ok" if route_dicts else "no_route",
            "n_routes": len(route_dicts),
            "summary": {
                "best_score": best_route["score"] if best_route else None,
                "best_solved_fraction": best_route["solved_fraction"] if best_route else 0.0,
                "best_n_steps": best_route["n_steps"] if best_route else 0,
            },
            "routes": route_dicts,
            "visualization": (
                best_route["visualization"]
                if best_route else {
                    "nodes": [],
                    "edges": [],
                    "layout": "tree",
                    "root_id": None,
                }
            ),
        }
