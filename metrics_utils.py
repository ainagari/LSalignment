
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import Counter
from scipy.spatial.distance import cosine



pronouns = ['I', 'you', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'you', 'your', 'yours', 'yourself', 'you', 'you', 'your', 'your', 'yourselves', 'he', 'him', 'his', 'his', 'himself', 'she', 'her', 'her', 'her', 'herself', 'it', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themself', 'they', 'them', 'their', 'theirs', 'themselves', "that", "this", "these", "those"]


def levenshteinDistanceDP(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
    
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]

def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(int(distances[t1][t2]), end=" ")
        print()



def shared_vocabulary_of_concept(mentions, sides):
    # first step: remove pronouns
    mentions = [x.lower() for x in mentions]
    mentions_nopron = [(x, s) for x, s in zip(mentions, sides) if x not in pronouns and s in ['for','against']]

    # there should be at least 3 left per side
    if len(mentions_nopron) < 2:
        return np.nan, np.nan

    fors = len([1 for x,s in mentions_nopron if s == "for"])
    againsts = len([1 for x,s in mentions_nopron if s == "against"])
    if fors < 3 or againsts < 3:
        return np.nan, np.nan

    dist_matrix = []
    for m in mentions_nopron:
        row = []
        for other_m in mentions_nopron:
            row.append(levenshteinDistanceDP(m[0], other_m[0]))
        dist_matrix.append(row)

    dist_array = np.array(dist_matrix)
    dist_array_condensed = squareform(dist_array)

    linkage_output = linkage(dist_array_condensed, method="average")

    clustering = fcluster(linkage_output, t=5, criterion="distance")


    assignments_by_side = {"for":[],"against":[]}
    for assignment, (mention, side) in zip(clustering, mentions_nopron):
        assignments_by_side[side].append(assignment)

    assignment_freqs = dict()
    for side in ['for','against']:
        assignment_freqs[side] = Counter(assignments_by_side[side])

    numerators = []
    for  c in assignment_freqs['for']:
        numerators.append(min(assignment_freqs['for'][c], assignment_freqs['against'][c]))

    numerator = sum(numerators)
    denominator = min(sum(assignment_freqs['for'].values()), sum(assignment_freqs['against'].values()))

    total_freq = sum(assignment_freqs['for'].values()) + sum(assignment_freqs['against'].values())

    overlap = numerator / denominator

    return overlap, total_freq



def driving_strength(asapp_a, asapp_b):
    '''a_t0: vector from side a at time 0
       b_t0: vector from side b at time 0
       a_t1, b_t1: same but at time 1 (the next time step)
    '''

    denominator = abs(asapp_a) + abs(asapp_b)

    DS_a = asapp_a / denominator
    DS_b = asapp_b / denominator

    return DS_a, DS_b





def SS_TU(all_reps_per_side): 
    ss_tu_values = {'cos':dict(), 'eucl':dict()}        
    for side in all_reps_per_side.keys():
        sims = []
        dists = []
        for i, rep in enumerate(all_reps_per_side[side]):
            for other_rep in all_reps_per_side[side][i+1:]:
                sims.append(1 - cosine(rep, other_rep))
                dists.append(np.linalg.norm(rep - other_rep))

        ss_tu_values['cos'][side] = np.average(sims)
        ss_tu_values['eucl'][side] = np.average(dists)
    return ss_tu_values


def OS_TU(all_reps_per_side):
    os_tu_values = dict()
    sims = []
    dists = []
    for i, rep in enumerate(all_reps_per_side['for']):
        for other_rep in all_reps_per_side['against']:
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))

    os_tu_values['cos'] = np.average(sims)
    os_tu_values['eucl'] = np.average(dists)

    return os_tu_values

    measures_here[mask]['TUOS_cos'] = np.average(sims)
    measures_here[mask]['TUOS_eucl'] = np.average(dists)



def SS_TA(cl, mask):    
    ss_ta_values = {'cos':dict(), 'eucl':dict()}    
    for side in cl[mask].keys():
        sims = []
        dists = []
        for i, rep in enumerate(cl[mask][side]['first-half']):
            rep = rep['representation']
            for other_rep in cl[mask][side]['second-half']:                            
                other_rep = other_rep['representation']
                sims.append(1 - cosine(rep, other_rep))
                dists.append(np.linalg.norm(rep - other_rep))                  

        ss_ta_values['cos'][side] = np.average(sims)
        ss_ta_values['eucl'][side] = np.average(dists)
    return ss_ta_values



def sApp(cl, mask):
    sims = []
    dists = []
    sApp = dict()
    for i, rep in enumerate(cl[mask]['for']['first-half']):
        rep = rep['representation']
        for other_rep in cl[mask]['against']['first-half']:                        
            other_rep = other_rep['representation']
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))
        sim_f1_a1 = np.average(sims)
        dist_f1_a1 = np.average(dists)

    sims = []
    dists = []
    for i, rep in enumerate(cl[mask]['for']['second-half']):
        rep = rep['representation']
        for other_rep in cl[mask]['against']['second-half']:
            other_rep = other_rep['representation']
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))
        sim_f2_a2 = np.average(sims)
        dist_f2_a2 = np.average(dists)              

    # with euclidean, switched positions so it is positive if they got closer
    sApp['cos'] = sim_f2_a2 - sim_f1_a1
    sApp['eucl'] = dist_f1_a1 - dist_f2_a2

    return sApp


def asApp(cl, mask):
    asap_values = {'cos':dict(), 'eucl':dict()}
    sims = []
    dists = []

    for i, rep in enumerate(cl[mask]['for']['first-half']):
        rep = rep['representation']
        for other_rep in cl[mask]['against']['first-half']:                        
            other_rep = other_rep['representation']
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))
        sim_f1_a1 = np.average(sims)
        dist_f1_a1 = np.average(dists)

    
    sims = []
    dists = []
    for i, rep in enumerate(cl[mask]['for']['second-half']):
        rep = rep['representation']
        for other_rep in cl[mask]['against']['first-half']:
            other_rep = other_rep['representation']
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))
        sim_f2_a1 = np.average(sims)
        dist_f2_a1 = np.average(dists) 


    sims = []
    dists = []
    for i, rep in enumerate(cl[mask]['against']['second-half']):
        rep = rep['representation']
        for other_rep in cl[mask]['for']['first-half']:
            other_rep = other_rep['representation']
            sims.append(1 - cosine(rep, other_rep))
            dists.append(np.linalg.norm(rep - other_rep))
        sim_f1_a2 = np.average(sims)
        dist_f1_a2 = np.average(dists) 



    # how much did for (A2) approach or get distant from against (B1) 
    # positive if approached
    asap_values['cos']['for'] = sim_f2_a1 - sim_f1_a1
    asap_values['cos']['against'] = sim_f1_a2 - sim_f1_a1              
    asap_values['eucl']['for'] = dist_f1_a1 - dist_f2_a1              
    asap_values['eucl']['against'] = dist_f1_a1 - dist_f1_a2

    return asap_values



def DS(asapp_a, asapp_b):    
    denominator = abs(asapp_a) + abs(asapp_b)

    DS_a = asapp_a / denominator
    DS_b = asapp_b / denominator

    return DS_a, DS_b
