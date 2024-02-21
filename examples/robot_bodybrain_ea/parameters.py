"""Experiment class."""

import sqlalchemy.orm as orm
from base import Base
from revolve2.experimentation.database import HasId
import config
import multineat


# the optimal multineat parameters for both the body and brain
def make_multineat_params() -> multineat.Parameters:
    multineat_params = multineat.Parameters()

    multineat_params.PopulationSize = 100
    multineat_params.TournamentSize = 5

    multineat_params.MutateRemLinkProb = 0.03# baseline 0.02
    multineat_params.RecurrentProb = 0.0
    multineat_params.OverallMutationRate = 0.15 #important
    multineat_params.MutateAddLinkProb = 0.1406 # baseline 0.08
    multineat_params.MutateAddNeuronProb = 0.1815# baseline 0.01
    multineat_params.MutateRemSimpleNeuronProb = 0.1137# baseline 0 TODO: check
    multineat_params.MutateNeuronBiasesProb = 0.0108 # baseline 0 or 0.7?

    multineat_params.MutateWeightsProb = 0.6827 # baseline 0.90
    multineat_params.MaxWeight = 8.0
    multineat_params.WeightMutationMaxPower = 0.2
    multineat_params.WeightReplacementMaxPower = 1.0
    multineat_params.MutateActivationAProb = 0.0
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0

    multineat_params.MutateNeuronActivationTypeProb = 0.0201  #baseline 0.03

    multineat_params.MutateOutputActivationFunction = False

    multineat_params.ActivationFunction_SignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_UnsignedSigmoid_Prob = 0.0
    multineat_params.ActivationFunction_Tanh_Prob = 1.0
    multineat_params.ActivationFunction_TanhCubic_Prob = 0.0
    multineat_params.ActivationFunction_SignedStep_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedStep_Prob = 0.0
    multineat_params.ActivationFunction_SignedGauss_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedGauss_Prob = 0.0
    multineat_params.ActivationFunction_Abs_Prob = 0.0
    multineat_params.ActivationFunction_SignedSine_Prob = 1.0
    multineat_params.ActivationFunction_UnsignedSine_Prob = 0.0
    multineat_params.ActivationFunction_Linear_Prob = 1.0

    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params

class EvolutionParameters:
    def __init__(
            self, database_file = config.DATABASE_FILE, num_repetitions = config.NUM_REPETITIONS, num_simulators = config.NUM_SIMULATORS, population_size = config.POPULATION_SIZE, offspring_size = config.OFFSPRING_SIZE, num_generations = config.NUM_GENERATIONS, tournament_size = config.TOURNAMENT_SIZE, sim_time = config.SIM_TIME
            ):
        self.database_file = database_file
        self.num_repetitions = num_repetitions
        self.num_simulators = num_simulators
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.sim_time = sim_time

    def __str__(self):
        return f"EvolutionParameters(database_file={self.database_file}, num_repetitions={self.num_repetitions}, num_simulators={self.num_simulators}, population_size={self.population_size}, offspring_size={self.offspring_size}, num_generations={self.num_generations}, tournament_size={self.tournament_size}, sim_time={self.sim_time})"

    def __repr__(self):
        return f"EvolutionParameters(database_file={self.database_file}, num_repetitions={self.num_repetitions}, num_simulators={self.num_simulators}, population_size={self.population_size}, offspring_size={self.offspring_size}, num_generations={self.num_generations}, tournament_size={self.tournament_size}, sim_time={self.sim_time})"
    



class ExperimentParameters(Base, HasId):
    """Experiment description."""

    __tablename__ = "parameters"

    # Evolution parameters
    evolution_parameters: orm.Mapped[EvolutionParameters] = orm.mapped_column(nullable=False)

    # multiNEAT parameters
    brain_multineat_parameters: orm.Mapped[multineat.Parameters] = orm.mapped_column(nullable=False)

    body_multineat_parameters: orm.Mapped[multineat.Parameters] = orm.mapped_column(nullable=False)

    def __init__(self):
        # Directly assign default values to instance attributes
        self.evolution_parameters = EvolutionParameters()  # Assuming EvolutionParameters() returns a default value
        self.brain_multineat_parameters = make_multineat_params()  # Assuming make_multineat_params() returns a default value
        self.body_multineat_parameters = make_multineat_params()  # Similarly, assuming this function returns a default value


    

    
