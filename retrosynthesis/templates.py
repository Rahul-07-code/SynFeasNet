from rdkit.Chem import AllChem

REACTIONS = {

    "ester_cleavage":
    AllChem.ReactionFromSmarts(
        "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[O:3][C:4]"
    ),

    "amide_cleavage":
    AllChem.ReactionFromSmarts(
        "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]"
    ),

    "ether_cleavage":
    AllChem.ReactionFromSmarts(
        "[O:1][C:2]>>[O:1].[C:2]"
    )
}