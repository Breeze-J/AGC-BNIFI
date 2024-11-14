import torch
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
from AGCBNIFI import AGCBNIFI
from utils import setup_seed, target_distribution, eva, LoadDataset, get_data
from opt import args
import datetime
from logger import Logger, metrics_info, record_info
import time
import random

# from visualization import t_sne
# import matplotlib.pyplot as plt


nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')

# result_folder = 'tsne_cite_results'
# os.makedirs(result_folder, exist_ok=True)

def fitness_function(a, b, c, d, ae_loss,loss_w, loss_a, KL_qp, KL_q1q, KL_q1p):
    loss = ae_loss + loss_w + a * loss_a + b * KL_qp + c * KL_q1q + d * KL_q1p
    return loss

def generate_initial_population(size, bounds):
    return [(
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[0], bounds[1]),
        random.uniform(bounds[0], bounds[1])
    ) for _ in range(size)]

def selection(population, fitnesses, num_parents):
    parents = []
    total_fitness = sum(fitnesses)
    probabilities = [1/(fitness + 1e-4) / total_fitness for fitness in fitnesses]  # 防止除以零
    for _ in range(num_parents):
        parent = random.choices(population, probabilities)[0]
        parents.append(parent)
    return parents

def crossover(parent1, parent2):
    child_a = ((parent1[0] + parent2[0]) / 2,
                (parent1[1] + parent2[1]) / 2,
                (parent1[2] + parent2[2]) / 2,
                (parent1[3] + parent2[3]) / 2)
    return child_a

def mutation(child, mutation_rate, bounds):
    if random.random() < mutation_rate:
        mutation_value = [random.uniform(-0.1, 0.1) for _ in range(4)]  # 调整变异幅度
        child = tuple(max(bounds[0], min(child[i] + mutation_value[i], bounds[1])) for i in range(4))
    return child

def genetic_algorithm(fitness_function, bounds, ae_loss, loss_w, loss_a, KL_qp, KL_q1q, KL_q1p, pop_size=200, generations=10, mutation_rate=0.1):
    population = generate_initial_population(pop_size, bounds)

    for generation in range(generations):
        fitnesses = [fitness_function(ind[0], ind[1], ind[2], ind[3], ae_loss, loss_w, loss_a, KL_qp, KL_q1q, KL_q1p) for ind in population]

        parents = selection(population, fitnesses, num_parents=pop_size // 2)

        next_generation = []
        while len(next_generation) < pop_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate, bounds)
            next_generation.append(child)

        population = next_generation


        fitnesses = [fitness_function(ind[0], ind[1], ind[2], ind[3], ae_loss, loss_w, loss_a, KL_qp, KL_q1q, KL_q1p) for ind in population]  # Recalculate fitnesses
        best_fitness = min(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")


    fitnesses = [fitness_function(ind[0], ind[1], ind[2], ind[3], ae_loss, loss_w, loss_a, KL_qp, KL_q1q, KL_q1p) for ind in population]  # Recalculate fitnesses
    best_fitness = min(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    return best_individual, best_fitness

def train(model, x, y):

    acc_reuslt = []
    nmi_result = []
    ari_result = []
    f1_result = []
    original_acc = -1
    metrics = [' acc', ' nmi', ' ari', ' f1']
    logger = Logger(args.name + '==' + nowtime)
    logger.info(model)
    logger.info(args)
    logger.info(metrics_info(metrics))

    n_clusters = args.n_clusters

    optimizer = Adam(model.parameters(), lr=args.lr)

    with torch.no_grad():
        z, _, _, _, _, _, _, _ = model.ae(x)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    logger.info("%s%s" % ('Initialization: ', record_info(eva(y, cluster_id))))


    for epoch in range(300):
        x_bar, z_hat, adj_hat, z_ae, q, q1, z_l = model(x, adj)
        if epoch % 1 == 0:

            tmp_q = q.data
            p = target_distribution(tmp_q)

        ae_loss = F.mse_loss(x_bar, x)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())

        KL_qp = F.kl_div(q.log(), p, reduction='batchmean')
        KL_q1q = F.kl_div(q1.log(), q, reduction='batchmean')
        KL_q1p = F.kl_div(q1.log(), p, reduction='batchmean')


        # best_individual, best_fitness = genetic_algorithm(fitness_function, bounds=(0, 1), re_loss=re_loss, KL_qp=KL_qp, KL_q1q=KL_q1q, KL_q1p=KL_q1p)
        # print(f"Optimal (a, b) = {best_individual}, with fitness = {best_fitness}")

        bounds = (0, 5)
        best_individual, best_fitness = genetic_algorithm(fitness_function, bounds, ae_loss, loss_w, loss_a, KL_qp, KL_q1q, KL_q1p, pop_size=50, generations=20, mutation_rate=0.2)
        print(f"Optimal parameters: a={best_individual[0]}, b={best_individual[1]}, c={best_individual[2]}, d={best_individual[3]}")
        print(f"Best fitness: {best_fitness}")
        a, b, c, d = best_individual

        loss = ae_loss + loss_w + a * loss_a + b * KL_qp + c * KL_q1q + d * KL_q1p

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        res = q1.data.cpu().numpy().argmax(1)
        acc, nmi, ari, f1 = eva(y, res, str(epoch) + 'Q1')

        logger.info("epoch%d%s:\t%s" % (epoch, ' Q1', record_info([acc, nmi, ari, f1])))

        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        f1_result.append(f1)

        if acc >= original_acc:
            original_acc = acc
            torch.save(model.state_dict(), './model_save/{}.pkl'.format(args.name))

    best_acc = max(acc_reuslt)
    t_nmi = nmi_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_ari = ari_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_f1 = f1_result[np.where(acc_reuslt == np.max(acc_reuslt))[0][0]]
    t_epoch = np.where(acc_reuslt == np.max(acc_reuslt))[0][0]
    logger.info("%sepoch%d:\t%s" % ('Best Acc is at ', t_epoch, record_info([best_acc, t_nmi, t_ari, t_f1])))

if __name__ == "__main__":
    setup_seed(2018)
    device = torch.device("cuda" if args.cuda else "cpu")
    x, y, adj = get_data(args.name)
    adj = adj.to(device)


    dataset = LoadDataset(x)
    x = torch.Tensor(dataset.x).to(device)

    model = AGCBNIFI(
        ae_n_enc_1=500,
        ae_n_enc_2=500,
        ae_n_enc_3=2000,
        ae_n_dec_1=2000,
        ae_n_dec_2=500,
        ae_n_dec_3=500,
        gae_n_enc_1=500,
        gae_n_enc_2=500,
        gae_n_enc_3=2000,
        gae_n_dec_1=2000,
        gae_n_dec_2=500,
        gae_n_dec_3=500,
        n_input= args.n_input,
        n_z= args.n_z,
        n_clusters=args.n_clusters,
        v=1.0).to(device)

    start_time = time.time()
    train(model, x, y)
