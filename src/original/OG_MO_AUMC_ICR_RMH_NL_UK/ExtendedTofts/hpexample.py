class simulation_hyper_parameters:
    def __init__(self):
        # simulation bounds
        self.sim = True
        self.rs_1_train = 123
        self.num_samples = 1000
        self.num_samples_leval = 1000
        self.vp_min = 0.001
        self.vp_max = 0.05  # was 0.02
        self.ve_min = 0.01
        self.ve_max = 0.7  # was 1
        self.kep_min = 0.1
        self.kep_max = 2.0
        self.R1_min = 1 / 2
        self.R1_max = 1 / 0.3
        self.plot = True
        self.S0 = 1000.0  # simulated S0


class acquisition_parameters:
    def __init__(self):
        self.r1 = 4.3  # relaxation of contrast agent
        self.TR = 7.2e-3  # repetition time in ms for the new toolbox
        self.FA1 = 4.0 / 180 * 3.14159  # FA in radian (low FA for T1 estimation)
        self.FA2 = 24 / 180 * 3.14159  # FA in radian (high FA = FA during DCE)
        self.time = 1.6  # time per "image"/ datapoint (seconds)
        self.rep1 = 12  # number of repeated low FA images
        self.rep2 = 100  # number of DCE images
        self.Tonset_min = self.time * (15)  # initialize 12+17=29 is mean onset time
        self.Tonset_max = self.time * (22)  # initialize


class Hyperparams:
    def __init__(self):
        """Hyperparameters"""
        self.simulations = simulation_hyper_parameters()
        self.acquisition = acquisition_parameters()
        self.lsq = True
        self.jobs = 8
        self.model = (
            "Cosine8"  # or Cosine4 --> Cosine8 is a more elaborate model for the AIF
        )
