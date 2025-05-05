# %%
# Helper functions to convert between the board and a column ,row tuple

# %%
from kaggle_environments import evaluate, make


def my_agent(observation, configuration):
    from random import choice

    SELECTED_COLUMN = 1

    ##return 0
    for n in range(configuration.columns):
        print(
            f"""
            *********************************************************
              Number of columns: {n}
              Number of rows: {configuration.rows}
              Total number of cells: {len(observation.board)}
              Number of Filled cells: {len([c for c in observation.board if c != 0])}
              -------------------------
              Board: {observation.board}
            *********************************************************
              """
        )

        return SELECTED_COLUMN

    # Not exactly what this for if 1 is always returned?
    return choice(
        [c for c in range(configuration.columns) if observation.board[c] == 0]
    )


def find_available_boards():
    pass


def get_value():
    pass


if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()

    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    env.render(mode="ipython", width=500, height=450)


# %%
