# coding: utf-8
"""
Code for evaluationg the effectiveness of TDGA that calculates entropy for each bi-allele.
This script is based on the TDGA implementation by Akira Terauchi.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from tdga.td_selection import ThermoDynamicalSelection
from utils.analyzer import Analyzer


def run_main(mut_rate, sel_type, Np, Ngen, t_init=10):
    """
    (always repuired)
    :param mut_rate: Mutation rate
    :param sel_type: Selection type ("sga" or "tdga" or "tdga_2")
    :param Np: The number of individuals
    :param Ngen: The number of generation

    (required if sel_type == "tdga" or "tdga_2")
    :param t_init: Initial temperature (defaulte: 10)
    """

    # Problem definition
    W = 744
    items = [
        (75, 7),
        (84, 9),
        (58, 13),
        (21, 5),
        (55, 16),
        (95, 28),
        (28, 15),
        (76, 43),
        (88, 60),
        (53, 37),
        (58, 44),
        (81, 63),
        (32, 34),
        (89, 95),
        (54, 61),
        (23, 29),
        (42, 57),
        (52, 72),
        (58, 83),
        (53, 84),
        (30, 48),
        (26, 45),
        (40, 74),
        (40, 78),
        (26, 52),
        (39, 79),
        (25, 64),
        (23, 64),
        (16, 55),
        (12, 74)
    ]
    optimum_val = 1099

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, len(items))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalation(individual):
        v_sum = 0
        w_sum = 0
        for i, x in enumerate(individual):
            v, w = items[i]
            v_sum += x * v
            w_sum += x * w
        return (v_sum,) if w_sum <= W else (0, )

    toolbox.register("evaluate", evalation)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mut_rate)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    if sel_type == "sga":
        toolbox.register("select", tools.selTournament, tournsize=3)
    elif sel_type == "tdga":
        tds = ThermoDynamicalSelection(
            Np=Np, t_init=t_init, scheduler=lambda x: x, is_bi_allele=False)
        toolbox.register("select", tds.select)
    elif sel_type == "tdga_2":
        tds = ThermoDynamicalSelection(
            Np=Np, t_init=t_init, scheduler=lambda x: x, is_bi_allele=True)
        toolbox.register("select", tds.select)

    pop = toolbox.population(n=Np)
    CXPB, MUTPB, NGEN = 1, 1, Ngen

    print("Start of evolution")

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("Evaluated %i individuals" % len(pop))

    analizer = Analyzer()
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        elite = tools.selBest(pop, 1)
        elite = list(map(toolbox.clone, elite))
        offspring = list(map(toolbox.clone, pop))

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values

        gen = pop + offspring  # 2Np
        for mutant in gen:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        gen += elite

        invalid_ind = [ind for ind in gen if not ind.fitness.valid]
        print("Evaluated %i individuals" % len(invalid_ind))
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        selected = toolbox.select(gen, k=Np)
        pop[:] = selected

        record = stats.compile(pop)

        print("  Min %s" % record["min"])
        print("  Max %s" % record["max"])
        print("  Avg %s" % record["avg"])
        print("  Std %s" % record["std"])

        analizer.add_pop(list(map(toolbox.clone, pop)))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    best_ind_fitness = best_ind.fitness.values[0]
    if best_ind_fitness == optimum_val:
        is_optimal = True
    else:
        is_optimal = False
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    analizer.plot_entropy_matrix(
        file_name="../results/" + sel_type + "_entropy.png")
    analizer.plot_stats(file_name="../results/" + sel_type +
                        "_stats.png", optimum_val=optimum_val)
    return best_ind, best_ind_fitness, is_optimal


def main():
    experiment_mut_rate = False
    experiment_np = False

    if experiment_mut_rate:
        # comparison of each method with various mutation rates.
        mut_rate_list = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        frequency_list = []
        for sel_type in ["tdga_2", "tdga", "sga"]:
            if sel_type == "sga":
                Np = 64
            else:
                Np = 32
            each_sel_type_frequency = []
            for mut_rate in mut_rate_list:
                tmp_freq = 0
                for i in range(30):
                    best_ind, best_ind_fitness, is_optimal = run_main(
                        mut_rate=mut_rate, Np=Np, Ngen=100, sel_type=sel_type, t_init=10)
                    if is_optimal:
                        tmp_freq += 1
                each_sel_type_frequency.append(tmp_freq)
            frequency_list.append(each_sel_type_frequency)

        # draw a graph
        plt.figure()
        plt.plot(mut_rate_list, frequency_list[0], color="black", marker="o", markersize=12,
                 markerfacecolor="black", linestyle="solid", label="SGA")
        plt.plot(mut_rate_list, frequency_list[1], color="black", marker="o", markersize=12,
                 markerfacecolor="gray", linestyle="dashed", label="TDGA")
        plt.plot(mut_rate_list, frequency_list[2], color="black", marker="o", markersize=12,
                 markerfacecolor="white", linestyle="dashdot", label="TDGA(bi-allele)")
        plt.xscale('log')
        plt.xlabel('Mutation Rate')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig('../results/mutation_rate.png')

    if experiment_np:
        # comparison of each method with various population sizes.
        Np_list = [8, 16, 32, 64, 128]
        frequency_list = []
        for sel_type in ["tdga_2", "tdga", "sga"]:
            if sel_type == "sga":
                mut_rate = 0.02
            elif sel_type == "tdga":
                mut_rate = 0.005
            elif sel_type == "tdga_2":
                mut_rate = 0.02
            each_Np_frequency = []
            for Np in Np_list:
                tmp_freq = 0
                for i in range(30):
                    best_ind, best_ind_fitness, is_optimal = run_main(
                        mut_rate=mut_rate, Np=Np, Ngen=100, sel_type=sel_type, t_init=10)
                    if is_optimal:
                        tmp_freq += 1
                each_Np_frequency.append(tmp_freq)
            frequency_list.append(each_Np_frequency)

        # draw a graph
        plt.figure()
        plt.plot(Np_list, frequency_list[0], color="black", marker="o", markersize=12,
                 markerfacecolor="black", linestyle="solid", label="SGA")
        plt.plot(Np_list, frequency_list[1], color="black", marker="o", markersize=12,
                 markerfacecolor="gray", linestyle="dashed", label="TDGA")
        plt.plot(Np_list, frequency_list[2], color="black", marker="o", markersize=12,
                 markerfacecolor="white", linestyle="dashdot", label="TDGA(bi-allele)")
        plt.xscale('log')
        plt.xlabel('Mutation Rate')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig('../results/population_size.png')

    # just run main
    # sga: mut_rate=0.005, Np=64
    # tdga, tdga_2: mut_rate=0.005, Np=32
    best_ind, best_ind_fitness, is_optimal = run_main(
        mut_rate=0.005, Np=64, Ngen=100, sel_type="sga", t_init=10)


if __name__ == "__main__":
    main()
