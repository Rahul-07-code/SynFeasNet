def score_route(
    spi_score,
    sa_score,
    scscore,
    syba_score,
    solved_fraction,
    n_steps
):

    return (
        0.40 * spi_score
        + 0.25 * syba_score
        - 0.15 * sa_score
        - 0.10 * scscore
        + 0.15 * solved_fraction
        - 0.05 * n_steps
    )