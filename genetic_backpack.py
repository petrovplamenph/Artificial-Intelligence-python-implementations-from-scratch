import pandas as pd
import random 
import numpy as np
labels = ['ITEM', 'WEIGHT[g]','VALUE']
data = [('map', 90 ,150),
('compass', 130, 35),
('water', 1530, 200),
('sandwich', 500, 160),
('glucose', 150, 60),
('tin', 680, 45),
('banana', 270, 60),
('apple', 390, 40),
('cheese', 230, 30),
('beer', 520, 10),
('suntan cream', 110, 70),
('camera', 320, 30),
('T-shirt', 240, 15),
('trousers', 480, 10),
('umbrella', 730, 40),
('waterproof trousers', 420, 70),
('waterproof overclothes', 430, 75),
('note-case', 220, 80),
('sunglasses', 70, 20),
('towel', 180, 12),
('socks', 40, 50),
('book', 300, 10),
('notebook', 900, 1),
('tent', 2000, 150)]
df = pd.DataFrame.from_records(data, columns=labels)
max_weight = 5000
best_elemet = []
best_score = 0

def create_array(column_of_interest, times_to_repeat):
    column_to_repeat = list(df[column_of_interest].values.reshape(1, df.shape[0])[0])
    repeated_col = np.array([column_to_repeat for dummy  in range(times_to_repeat)])
    return repeated_col


def fitness(parents, weights, values):
    """fitness function of the popul"""
    backpack_weights = np.sum(np.multiply(weights, parents), axis=1)
    boolian_weights = backpack_weights > max_weight
    for idx in range(len(backpack_weights)):
        if boolian_weights[idx]:
            items_idx = np.where(parents[idx])[0]
            np.random.shuffle(items_idx)
            i = 0
            while backpack_weights[idx] > max_weight:

                parents[idx][items_idx[i]] = 0
                backpack_weights[idx] -= weights[0][items_idx[i]]
                i += 1
    fit = np.sum(np.multiply(values, parents), axis=1)
    fit_sorted = fit.argsort()
    fit = fit[fit_sorted[::-1]]
    parents = parents[fit_sorted[::-1]]
    fitns_eval = np.std(fit[:int(len(fit)*0.8)])
    # store the best backpack itmes in the variable best_element 
    global best_elemet, best_score
    if fit[0] > best_score:
        best_score = fit[0]
        best_elemet = parents[0]
    return (parents, fitns_eval)


def mutation(parents, mut_to_0, n_mut_to_1, number_of_elements):
    np.random.shuffle(mut_to_0)
    mut_to_0 = mut_to_0.reshape(number_of_elements, df.shape[0])
    mut_rows = np.random.randint(df.shape[0], size = n_mut_to_1)
    mut_cols = np.random.randint(number_of_elements, size = n_mut_to_1)
    parents = np.multiply(mut_to_0, parents)
    for row,col in zip(mut_rows,mut_cols):
        parents[col][row] = 1
    return parents


def get_childs(parents, cites, masks, number_of_elements):
    
    new_childs = np.zeros((number_of_elements, df.shape[0]))
    space = int(number_of_elements/4)
    tottal_chosen = []
    np.random.shuffle(cites)
    for gen in range(4):
        for idx in range(len(masks)):
            np.random.shuffle(masks[idx])
        group_1 = parents[:space,:][masks[0]]
        group_2 = parents[space:int(space*2),:][masks[1]]
        group_3 = parents[int(2*space):int(3*space),:][masks[2]]
        group_4 = parents[int(3*space):,:][masks[3]]
        chosen_parents = np.concatenate((group_1, group_2,group_3,group_4), axis=0) 
        np.random.shuffle(chosen_parents)

        tottal_chosen.append(chosen_parents)
    parents = np.concatenate((tottal_chosen[0], tottal_chosen[1],tottal_chosen[2],tottal_chosen[3][:150,:]), axis=0)
    end_cond = int(space*3+150)
    idx = 0
    while idx != end_cond:
        new_childs[idx,:] =np.append(parents[idx,:cites[idx]], parents[idx+1,cites[idx]:])
        parents[idx+1,:] 
        new_childs[idx+1,:] = np.append(parents[idx+1,:cites[idx]], parents[idx,cites[idx]:])
        idx += 2
    par_idx = 0
    for idx in range(900,1000):
        new_childs[idx,:] = np.logical_xor(chosen_parents[par_idx,:],chosen_parents[-par_idx,:])
        par_idx += 1
    return new_childs


def choose_init_elements(number_of_elements):
    elements_idx = list(df.index)
    childs = np.zeros((number_of_elements, len(elements_idx)))
    weights = df['WEIGHT[g]']
    for row in range(number_of_elements):
        curr_weight = 0
        random.shuffle(elements_idx)
        for idx in elements_idx:
            curr_weight += weights[idx]
            if  curr_weight > max_weight:
                break
            childs[row][idx] = 1
    return childs

      
def create_masks(number_of_elements):
    space = int(number_of_elements/4)
    mask1 = np.zeros(space, dtype=bool)
    mask1[:int(space*0.5)] = True
    mask2 = np.zeros(space, dtype=bool)
    mask2[:int(space*0.3)] = True
    mask3 = np.zeros(space, dtype=bool)
    mask3[:int(space*0.14)] = True
    mask4 = np.zeros(space, dtype=bool)
    mask4[:int(space*0.06)] = True
    return[mask1, mask2, mask3, mask4]

  
def main():
    number_of_elements = 1000
    percent_mut = 2
    n_mut_to_0 = int(number_of_elements * percent_mut * df.shape[0]/100)
    n_mut_to_1 = int(n_mut_to_0 * 1.05)
    mut_to_0 = np.ones(number_of_elements*df.shape[0])
    mut_to_0[:n_mut_to_0] = 0
    cites = np.array([cite for cite in range(1,df.shape[0]) for dummy in range(number_of_elements)])
    add_cites = np.random.randint(1, df.shape[0]-1, size = number_of_elements)
    cites = np.concatenate((cites, add_cites), axis=0)
    masks = create_masks(number_of_elements)
    max_itter  = 1000
    weights = create_array('WEIGHT[g]', number_of_elements)
    values = create_array('VALUE', number_of_elements)
    parents = choose_init_elements(number_of_elements)
    parents = mutation(parents, mut_to_0, n_mut_to_1, number_of_elements)
    parents, std_80 = fitness(parents, weights, values)
    parents = get_childs(parents,cites,masks,number_of_elements)
    for dummy in range(max_itter):
        parents = mutation(parents, mut_to_0, n_mut_to_1, number_of_elements)
        parents, std_80 = fitness(parents, weights, values)
        current_best = parents[0]
        parents = get_childs(parents,cites,masks,number_of_elements)
        if std_80 < 70:
            break

        
    print('Best score:', best_score)
    print('Best element:', best_elemet)
    print('Items to take:',df['ITEM'][best_elemet==1])
    print()
    print('Best score from last generation:', best_score)
    print('Best elementfrom last generation:', best_elemet)
    print('Items to take:',df['ITEM'][current_best==1])

    
    
    
if __name__ == '__main__':
    main()