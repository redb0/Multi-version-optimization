class AlgorithmParams:
    def __init__(self, num_countries, num_initial_imperialists, z, rev_rate, damp_rat, assim_coef, stop_one_empire, threshold):
        self.num_of_initial_imperialists = num_initial_imperialists
        self.num_of_countries = num_countries
        self.zeta = z
        self.revolution_rate = rev_rate
        self.damp_ratio = damp_rat
        self.assimilation_coefficient = assim_coef
        self.stop_if_just_one_empire = stop_one_empire
        self.uniting_threshold = threshold
