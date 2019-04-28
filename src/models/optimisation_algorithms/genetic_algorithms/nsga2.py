#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
import sys
import matplotlib.pyplot as plt
import numpy
import array
import random
import json
import pandas as pd
from pathlib import Path
from scoop import futures
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

project_dir = Path("__file__").resolve().parents[1]
sys.path.insert(0, '{}/temporal_granularity/'.format(project_dir))

# import json

import numpy

# from math import sqrt

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools
from src.models.env.som_env import SOMEnv
from src.models.env.k_means_env import KMeansEnv
import logging

logging.basicConfig(level=logging.INFO)


year_start = "2006"
year_end = "2011"

onshore_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

onshore_data_np = onshore_data[(onshore_data.date > "2006") & (
    onshore_data.date < "2011")].reset_index().drop(
    columns=["date", 'index']).values

load_data = pd.read_csv(
    "{}/temporal_granularity/data/processed/data_grouped_by_day/load_NG_normalised_each_day.csv".format(project_dir))

load_data_np = load_data[(load_data.date > "2006") & (
    load_data.date < "2011")].reset_index().drop(columns=["date", 'index']).values


# offshore_data = pd.read_csv(
# '{}/temporal_granularity/data/processed/resources/offshore_processed.csv'.format(project_dir))
pv_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/data_grouped_by_day/pv_each_day.csv'.format(project_dir))

pv_data_np = pv_data[(pv_data.date > "2006") & (
    pv_data.date < "2011")].reset_index().drop(columns=["date", 'index']).values

pv_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/resources/pv_processed.csv'.format(project_dir))
onshore_data = pd.read_csv(
    '{}/temporal_granularity/data/processed/resources/onshore_processed.csv'.format(project_dir))
load_data = pd.read_csv(
    "{}/temporal_granularity/data/processed/demand/load_processed_normalised.csv".format(project_dir))


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", array.array, typecode='d',
               fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Problem definition
# Functions zdt1, zdt2, zdt3, zdt6 have bounds [0, 1]
BOUND_LOW, BOUND_UP = 0.0, 100

# Functions zdt4 has bounds x1 = [0, 1], xn = [-5, 5], with n = 2, ..., 10
# BOUND_LOW, BOUND_UP = [0.0] + [-5.0]*9, [1.0] + [5.0]*9

# Functions zdt1, zdt2, zdt3 have 30 dimensions, zdt4 and zdt6 have 10
NDIM = 3 * 12 * 12 + 2
# NDIM = 3 * 51 + 1


def uniform(low, up, size=None):
    try:
        return [random.randint(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.randint(a, b) for a, b in zip([low] * size, [up] * size)]


def evalMinSOM(individual):
    individual = [int(i) for i in individual]
    env = SOMEnv(pv_data_np, onshore_data_np, load_data_np,
                 pv_data, onshore_data, load_data, round(individual[0] / 10) + 2, round(individual[1] / 10) + 2, 20000, int(year_end))
    result = env.step(individual[2:])
    result = result[0], result[1], result[2]

    return result
    # return individual[0], individual[1], individual[2]


def evalMinKMeans(individual):
    individual = [int(i) for i in individual]
    env = KMeansEnv(pv_data_np, onshore_data_np, load_data_np,
                    pv_data, onshore_data, load_data, int(individual[0] / 2) + 1)
    result = env.step(individual[1:])
    result = result[0], result[1], result[2]
    return result


toolbox.register("map_distributed", futures.map)


toolbox.register("attr_int", np.random.randint, low=0, high=100)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_int, NDIM)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", evalMinKMeans)
toolbox.register("evaluate", evalMinSOM)
toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0 / NDIM)
toolbox.register("select", tools.selNSGA2)


def main(seed=None):
    random.seed(seed)

    NGEN = 250
    MU = 100
    CXPB = 0.9

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map_distributed(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):

        print("-- Generation %i --" % gen)

        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map_distributed(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

        best_ind = tools.selBest(pop, 1)[0]

        # print("Best individual is %s, %s" %
        #   (np.rint(best_ind), best_ind.fitness.values))

        front = numpy.array(
            [ind.fitness.values + tuple(ind) for ind in pop])

        # np.savetxt('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/k_means/data/pareto_front_{}.csv'.format(project_dir, gen), front, delimiter=",")
        np.savetxt('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/long_term/data/pareto_front_{}.csv'.format(
            project_dir, gen), front, delimiter=",")
        fig = plt.figure(1)

        columns = 2
        rows = 1

        fig.add_subplot(rows, columns, 1)
        plt.scatter(front[:, 0], front[:, 1], c="b")

        fig.add_subplot(rows, columns, 2)
        plt.scatter(front[:, 1], front[:, 2], c="b")

        # plt.savefig('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/k_means/images/pareto_front_{}.png'.format(project_dir, gen))
        plt.savefig('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/long_term/images/pareto_front_{}.png'.format(project_dir, gen))
        plt.close()

        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.scatter(front[:, 0], front[:, 1], front[:, 2], c='red')

        ax.axis("tight")
        fig.savefig('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/long_term/images/pareto_front_3D_{}.png'.format(project_dir, gen))
        # fig.savefig('{}/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front/k_means/images/pareto_front_3D_{}.png'.format(project_dir, gen))
        plt.close()

    # print("Final population hypervolume is %f" %
    #   hypervolume(pop, [11.0, 11.0]))

    return pop, logbook


if __name__ == "__main__":

    pop, stats = main()
    pop.sort(key=lambda x: x.fitness.values)

    print(stats)

    front = numpy.array([ind.fitness.values for ind in pop])
    plt.scatter(front[:, 0], front[:, 1], c="b")
    plt.axis("tight")
    plt.savefig('/Users/b1017579/Documents/PhD/Projects/14-temporal-granularity/temporal_granularity/src/models/optimisation_algorithms/genetic_algorithms/pareto_front.png')
