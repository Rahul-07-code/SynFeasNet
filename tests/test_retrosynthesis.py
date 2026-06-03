from retrosynthesis import RetrosynthesisEngine


def print_tree(
        node,
        indent=0
):

    print(
        " " * indent +
        node.smiles
    )

    for child in node.children:

        print_tree(
            child,
            indent + 4
        )


engine = RetrosynthesisEngine()

routes = engine.run(
    "CC(=O)Oc1ccccc1C(=O)O"
)

print(
    f"Routes Found: {len(routes)}"
)

for idx, route in enumerate(
        routes
):

    print(
        f"\nRoute {idx+1}"
    )

    print(
        "Score:",
        route.score
    )

    print(
        "Solved Fraction:",
        route.solved_fraction
    )

    print_tree(
        route.root
    )