from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RouteNode:
    smiles: str
    depth: int = 0
    reaction: Optional[str] = None
    is_building_block: bool = False
    children: List["RouteNode"] = field(default_factory=list)

@dataclass
class RetrosynthesisRoute:
    root: RouteNode
    score: float = 0.0
    solved_fraction: float = 0.0
    n_steps: int = 0


def route_node_to_dict(node: RouteNode) -> Dict[str, Any]:
    return {
        "smiles": node.smiles,
        "depth": node.depth,
        "reaction": node.reaction,
        "is_building_block": node.is_building_block,
        "is_leaf": len(node.children) == 0,
        "children": [
            route_node_to_dict(child)
            for child in node.children
        ],
    }


def route_to_visualization(route: RetrosynthesisRoute) -> Dict[str, Any]:
    nodes = []
    edges = []

    def visit(node: RouteNode, node_id: str, parent_id: Optional[str] = None):
        nodes.append({
            "id": node_id,
            "smiles": node.smiles,
            "label": node.smiles,
            "depth": node.depth,
            "type": "building_block" if node.is_building_block else "intermediate",
            "is_building_block": node.is_building_block,
            "is_leaf": len(node.children) == 0,
        })

        if parent_id is not None:
            edges.append({
                "source": parent_id,
                "target": node_id,
                "reaction": node.reaction,
            })

        for index, child in enumerate(node.children):
            visit(child, f"{node_id}.{index}", node_id)

    visit(route.root, "0")

    return {
        "nodes": nodes,
        "edges": edges,
        "layout": "tree",
        "root_id": "0",
    }


def route_to_dict(route: RetrosynthesisRoute, rank: int = 1) -> Dict[str, Any]:
    return {
        "rank": rank,
        "score": round(float(route.score), 4),
        "solved_fraction": round(float(route.solved_fraction), 4),
        "n_steps": int(route.n_steps),
        "tree": route_node_to_dict(route.root),
        "visualization": route_to_visualization(route),
    }
