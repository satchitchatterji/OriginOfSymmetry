from sqlalchemy.types import TypeDecorator, VARCHAR
import json



import sqlalchemy.orm as orm
from base import Base
from revolve2.experimentation.database import HasId
import config
import multineat

def serialize_multineat_parameters(params):
    # Convert a multineat.Parameters object to a dictionary
    data = {
        "PopulationSize": params.PopulationSize,
        "TournamentSize": params.TournamentSize,
        "MutateRemLinkProb": params.MutateRemLinkProb,
        "RecurrentProb": params.RecurrentProb,
        "OverallMutationRate": params.OverallMutationRate,
        "MutateAddLinkProb": params.MutateAddLinkProb,
        "MutateAddNeuronProb": params.MutateAddNeuronProb,
        "MutateRemSimpleNeuronProb": params.MutateRemSimpleNeuronProb,
        "MutateNeuronBiasesProb": params.MutateNeuronBiasesProb,
        "MutateWeightsProb": params.MutateWeightsProb,
        "MaxWeight": params.MaxWeight,
        "WeightMutationMaxPower": params.WeightMutationMaxPower,
        "WeightReplacementMaxPower": params.WeightReplacementMaxPower,
        "MutateActivationAProb": params.MutateActivationAProb,
        "ActivationAMutationMaxPower": params.ActivationAMutationMaxPower,
        "MinActivationA": params.MinActivationA,
        "MaxActivationA": params.MaxActivationA,
        "MutateNeuronActivationTypeProb": params.MutateNeuronActivationTypeProb,
        "MutateOutputActivationFunction": params.MutateOutputActivationFunction,
        "ActivationFunction_SignedSigmoid_Prob": params.ActivationFunction_SignedSigmoid_Prob,
        "ActivationFunction_UnsignedSigmoid_Prob": params.ActivationFunction_UnsignedSigmoid_Prob,
        "ActivationFunction_Tanh_Prob": params.ActivationFunction_Tanh_Prob,
        "ActivationFunction_TanhCubic_Prob": params.ActivationFunction_TanhCubic_Prob,
        "ActivationFunction_SignedStep_Prob": params.ActivationFunction_SignedStep_Prob,
        "ActivationFunction_UnsignedStep_Prob": params.ActivationFunction_UnsignedStep_Prob,
        "ActivationFunction_SignedGauss_Prob": params.ActivationFunction_SignedGauss_Prob,
        "ActivationFunction_UnsignedGauss_Prob": params.ActivationFunction_UnsignedGauss_Prob,
        "ActivationFunction_Abs_Prob": params.ActivationFunction_Abs_Prob,
        "ActivationFunction_SignedSine_Prob": params.ActivationFunction_SignedSine_Prob,
        "ActivationFunction_UnsignedSine_Prob": params.ActivationFunction_UnsignedSine_Prob,
        "ActivationFunction_Linear_Prob": params.ActivationFunction_Linear_Prob,
        "MutateNeuronTraitsProb": params.MutateNeuronTraitsProb,
        "MutateLinkTraitsProb": params.MutateLinkTraitsProb,
        "AllowLoops": params.AllowLoops,
    }
    return data

def deserialize_multineat_parameters(data):
    # Create a multineat.Parameters object from a dictionary
    params = multineat.Parameters()

    params.PopulationSize = data["PopulationSize"]
    params.TournamentSize = data["TournamentSize"]
    params.MutateRemLinkProb = data["MutateRemLinkProb"]
    params.RecurrentProb = data["RecurrentProb"]
    params.OverallMutationRate = data["OverallMutationRate"]
    params.MutateAddLinkProb = data["MutateAddLinkProb"]
    params.MutateAddNeuronProb = data["MutateAddNeuronProb"]
    params.MutateRemSimpleNeuronProb = data["MutateRemSimpleNeuronProb"]
    params.MutateNeuronBiasesProb = data["MutateNeuronBiasesProb"]
    params.MutateWeightsProb = data["MutateWeightsProb"]
    params.MaxWeight = data["MaxWeight"]
    params.WeightMutationMaxPower = data["WeightMutationMaxPower"]
    params.WeightReplacementMaxPower = data["WeightReplacementMaxPower"]
    params.MutateActivationAProb = data["MutateActivationAProb"]
    params.ActivationAMutationMaxPower = data["ActivationAMutationMaxPower"]
    params.MinActivationA = data["MinActivationA"]
    params.MaxActivationA = data["MaxActivationA"]
    params.MutateNeuronActivationTypeProb = data["MutateNeuronActivationTypeProb"]
    params.MutateOutputActivationFunction = data["MutateOutputActivationFunction"]
    params.ActivationFunction_SignedSigmoid_Prob = data["ActivationFunction_SignedSigmoid_Prob"]
    params.ActivationFunction_UnsignedSigmoid_Prob = data["ActivationFunction_UnsignedSigmoid_Prob"]
    params.ActivationFunction_Tanh_Prob = data["ActivationFunction_Tanh_Prob"]
    params.ActivationFunction_TanhCubic_Prob = data["ActivationFunction_TanhCubic_Prob"]
    params.ActivationFunction_SignedStep_Prob = data["ActivationFunction_SignedStep_Prob"]
    params.ActivationFunction_UnsignedStep_Prob = data["ActivationFunction_UnsignedStep_Prob"]
    params.ActivationFunction_SignedGauss_Prob = data["ActivationFunction_SignedGauss_Prob"]
    params.ActivationFunction_UnsignedGauss_Prob = data["ActivationFunction_UnsignedGauss_Prob"]
    params.ActivationFunction_Abs_Prob = data["ActivationFunction_Abs_Prob"]
    params.ActivationFunction_SignedSine_Prob = data["ActivationFunction_SignedSine_Prob"]
    params.ActivationFunction_UnsignedSine_Prob = data["ActivationFunction_UnsignedSine_Prob"]
    params.ActivationFunction_Linear_Prob = data["ActivationFunction_Linear_Prob"]
    params.MutateNeuronTraitsProb = data["MutateNeuronTraitsProb"]
    params.MutateLinkTraitsProb = data["MutateLinkTraitsProb"]
    params.AllowLoops = data["AllowLoops"]

    return params

class JSONEncodedDict(TypeDecorator):
    """Enables JSON storage by encoding and decoding on the fly."""
    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value, default=lambda o: o.__dict__)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
            value = EvolutionParameters(**value)
        return value
    

class JSONEncodedMultineatParameters(TypeDecorator):
    """Enables JSON storage by encoding and decoding on the fly."""
    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(serialize_multineat_parameters(value))
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = deserialize_multineat_parameters(json.loads(value))
        return value



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
            self, database_file = config.DATABASE_FILE, num_repetitions = config.NUM_REPETITIONS, num_simulators = config.NUM_SIMULATORS, population_size = config.POPULATION_SIZE, offspring_size = config.OFFSPRING_SIZE, num_generations = config.NUM_GENERATIONS, tournament_size = config.TOURNAMENT_SIZE, sim_time = config.SIM_TIME,steer = False
            ):
        self.database_file = database_file
        self.num_repetitions = num_repetitions
        self.num_simulators = num_simulators
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.sim_time = sim_time
        self.steer = steer

    def __str__(self):
        return f"EvolutionParameters(database_file={self.database_file}, num_repetitions={self.num_repetitions}, num_simulators={self.num_simulators}, population_size={self.population_size}, offspring_size={self.offspring_size}, num_generations={self.num_generations}, tournament_size={self.tournament_size}, sim_time={self.sim_time}), steer={self.steer})"

    def __repr__(self):
        return f"EvolutionParameters(database_file={self.database_file}, num_repetitions={self.num_repetitions}, num_simulators={self.num_simulators}, population_size={self.population_size}, offspring_size={self.offspring_size}, num_generations={self.num_generations}, tournament_size={self.tournament_size}, sim_time={self.sim_time}), steer={self.steer})"
    



class ExperimentParameters(Base, HasId):
    """Experiment description."""

    __tablename__ = "parameters"
    # Assuming EvolutionParameters and multineat.Parameters can be serialized to JSON
    evolution_parameters: orm.Mapped[EvolutionParameters] = orm.mapped_column(JSONEncodedDict, nullable=False)
    brain_multineat_parameters: orm.Mapped["multineat.Parameters"] = orm.mapped_column(JSONEncodedMultineatParameters, nullable=False)
    body_multineat_parameters: orm.Mapped["multineat.Parameters"] = orm.mapped_column(JSONEncodedMultineatParameters, nullable=False)


    def __init__(self):
        # Directly assign default values to instance attributes
        self.evolution_parameters = EvolutionParameters()  # Assuming EvolutionParameters() returns a default value
        self.brain_multineat_parameters = make_multineat_params()  # Assuming make_multineat_params() returns a default value
        self.body_multineat_parameters = make_multineat_params()  # Similarly, assuming this function returns a default value


    

    
