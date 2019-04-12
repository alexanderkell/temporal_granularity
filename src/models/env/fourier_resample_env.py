from src.models.env.env import Env


class FourierEnv(Env):

    def __init__(self, solar, onshore, load, solar_df, onshore_df, load_df, n_days):
        super().__init__(solar, onshore, load, solar_df, onshore_df, load_df)

        self.n_days = n_days

        
