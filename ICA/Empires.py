class Empires:

    def __init__(self, imp_pos, imp_fit, colonies_pos, colonies_fit, total_fit):
        self.imperialist_position = imp_pos
        self.imperialist_fitness = imp_fit
        self.colonies_position = colonies_pos
        self.colonies_fitness = colonies_fit

        # ценность всей империи
        self.total_fitness = total_fit

