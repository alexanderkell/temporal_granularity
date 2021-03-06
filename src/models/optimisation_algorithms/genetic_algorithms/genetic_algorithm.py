import sys
from pathlib import Path
project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

import pandas as pd
from src.models.env.som_env import SOMEnv

"""
 Description: Use of genetic algorithm to optimally select representative days.
 Created on Fri Apr 05 2019

 Copyright (c) 2019 Newcastle University
 License is MIT
 Email is alexander@kell.es
"""

import random

from deap import base
from deap import creator
from deap import tools
from src.models.env.som_env import SOMEnv
import numpy as np
import logging
from src.models.manipulations.duration_curves import get_group_ldc

from src.metrics.multi_year_metrics import MultiYearMetrics

from scoop import futures

logger = logging.getLogger(__name__)


class GeneticAlgorithm:

    def __init__(self, pv_data, onshore_data, load_data, onshore_wide, load_wide, pv_wide, year_start):
        self.pv_data = pv_data
        self.onshore_data = onshore_data
        self.load_data = load_data

        self.onshore_wide = onshore_wide
        self.load_wide = load_wide
        self.pv_wide = pv_wide

        self.year_start = year_start


    def initialise(self):
        creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        # Attribute generator
        #                      define 'attr_bool' to be an attribute ('gene')
        #                      which corresponds to integers sampled uniformly
        #                      from the range [0,1] (i.e. 0 or 1 with equal
        #                      probability)
        toolbox.register("attr_bool", np.random.randint, low=0, high=100)

        # Structure initializers
        #                         define 'individual' to be an individual
        #                         consisting of 100 'attr_bool' elements ('genes')
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 3 * 11 * 11 + 2)

        # define the population to be a list of individuals
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # the goal ('fitness') function to be maximized

# onshore_data = pd.read_csv(
    # '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

        # onshore_data_np = self.onshore_data.reset_index().drop(columns=['index']).values

        # load_data = pd.read_csv("{}/temporal_granularity/data/processed/data_grouped_by_day/load_normalised_each_day.csv".format(project_dir))

        # load_data_np = self.load_data.reset_index().drop(columns=['index']).values


# offshore_data = pd.read_csv(
# '{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir))
        # pv_data = pd.read_csv('{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

        # pv_data_np = self.pv_data.reset_index().drop(columns=['index']).values

# pv_data = pd.read_csv(
#     '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
# onshore_data = pd.read_csv(
#     '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
# load_data = pd.read_csv(
#     "{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv".format(project_dir))

        #
        #
        # ----------
        # Operator registration
        # ----------
        # register the goal / fitness function
        toolbox.register("evaluate", self.evalOneMax)
        #
        # register the crossover operator
        toolbox.register("mate", tools.cxTwoPoint)
        #
        # register a mutation operator with a probability to
        # flip each attribute/gene of 0.05
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

        # operator for selecting individuals for breeding the next
        # generation: each individual of the current generation
        # is replaced by the 'fittest' (best) of three individuals
        # drawn randomly from the current generation.
        toolbox.register("select", tools.selTournament, tournsize=3)

        toolbox.register("map_distributed", futures.map)

        return toolbox

    def evalOneMax(self, individual):
        
        pv_data_np = self.pv_wide.reset_index().drop(columns=['date', 'index']).values
        onshore_data_np = self.onshore_wide.reset_index().drop(columns=['date', 'index']).values
        load_data_np = self.load_wide.reset_index().drop(columns=['date', 'index']).values

        env = SOMEnv(pv_data_np, onshore_data_np, load_data_np, self.pv_data, self.onshore_data, self.load_data, round(individual[0] / 10) + 1, round(individual[1] / 10) + 1, 20000, self.year_start)
        # logger.debug("individual : {}".format(individual))
        result = env.step(individual[2:])
        # logger.info("individual: {}, result: {}".format(individual, result))
        result = result[0], result[1], result[2]
        # return

        return result

    def run_genetic_algorithm(self, toolbox):
        random.seed(64)

        # create an initial population of 300 individuals (where
        # each individual is a list of integers)
        pop = toolbox.population(n=300)

        # CXPB  is the probability with which two individuals
        #       are crossed
        #
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.5, 0.2

        print("Start of evolution")

        # Evaluate the entire population
        fitnesses = list(toolbox.map_distributed(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of
        fits = [ind.fitness.values[0] for ind in pop]

        # Variable keeping track of the number of generations
        g = 0

        # Begin the evolution
        # while max(fits) < -5 and g < 1000:
        while g < 1000:
            # A new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)

                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:

                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            print("-- End of (successful) evolution --")

            best_ind = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
