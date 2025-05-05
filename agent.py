# %%
from kaggle_environments import evaluate, make

def my_agent(observation, configuration):
    from random import choice
    print(configuration.columns)

    ##return 0 
    for n in range(configuration.columns):
        print(n)
        print(observation.board)
        print(len(observation.board))
        print(type(observation.board))
        print(configuration.columns)
        print(observation.board[n])
        print(observation.board[n] == 0)

        
        return 1



    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

def find_available_boards():
    pass

def get_value():
    pass





if __name__ == "__main__":
    env= make("connectx", debug=True)
    env.render()

    env.reset()
    # Play as the first agent against default "random" agent.
    env.run([my_agent, "random"])
    env.render(mode="ipython", width=500, height=450)   


# %%
