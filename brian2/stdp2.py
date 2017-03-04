from brian2 import *
import brian2 as b
import brian2.numpy_ as np
from brian2tools import *
import os.path
from struct import unpack
import cPickle as pickle
import time
import numpy as np
import scipy 
import matplotlib.cm as cmap
from matplotlib.pyplot import *
from numpy.matlib import rand


# specify the location of the MNIST data
MNIST_data_path = ''

#------------------------------------------------------------------------------ 
# functions
#------------------------------------------------------------------------------  
def get_labeled_data(picklename, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename))
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]
    
        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in xrange(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in xrange(cols)]  for unused_row in xrange(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]
            
        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        #else:
            #n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    #else:
       # n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr

def get_2d_input_weights():
    name = 'XeAe'
    weight_matrix = np.zeros((n_input, n_e))
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
#     connMatrix = connections[name][:]
    connMatrix = synapses[name].w[:]
    weight_matrix = np.copy(connMatrix.reshape(n_input, n_e))
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights

def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = figure(fig_num, figsize = (5, 5))
    im2 = imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    b.colorbar(im2)
    b.title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def normalize_weights():
    synap = synapses[connName].w[:, :]  #need test
    temp_conn = np.copy(synap)
    colSums = np.sum(temp_conn, axis = 0)
    colFactors = weight['ee_input'] / colSums
#     for j in xrange(n_e):
#         synapses[connName].w[:, j] *= colFactors[j]
    synapses[connName].w[:] *= colFactors;

def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e    
    for j in xrange(10):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        for i in xrange(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

def update_2d_input_weights(im, fig):
    weights = get_2d_input_weights()
    im.set_array(weights)
    fig.canvas.draw()
    return im


def save_theta(ending = ''):
    print 'save theta'
    for pop_name in population_names:
        np.save(data_path + 'weights/theta_' + pop_name + ending, neuron_groups[pop_name + 'e'].theta)

def get_recognized_number_ranking(assignments, spike_rates):
    summed_rates = [0] * 10
    num_assignments = [0] * 10
    for i in xrange(10):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]
    return np.argsort(summed_rates)[::-1]


#------------------------------------------------------------------------------ 
# load MNIST data
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start
 

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
np.random.seed(0)

ending = ''
n_input = 784
n_e = 1
single_example_time =   0.35 * second
resting_time = 0.15 * second

v_rest_e = -65. * mV 
v_rest_i = -60. * mV 
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms

weight = {}
delay = {}

weight['ee_input'] = 78.
delay['ee_input'] = 10 * ms
delay['ei_input'] = 5 * ms

wmax_ee = 1.0
#valuable of stdp
tc_pre_ee    = 20 * ms
tc_post_1_ee = 20 * ms
tc_post_2_ee = 40 * ms
nu_ee_pre0    = 0.0001   # learning rate
nu_ee_post0   = 0.01     # learning rate
wmax_ee      = 1.0
exp_ee_pre   = 0.2
exp_ee_post  = exp_ee_pre
STDP_offset  = 0.4

tc_theta = 1e7 * ms
theta_plus_e = 0.05 * mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

###function of e neuron
neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS * -v                           : amp
        I_synI = gi * nS * (-100. * mV - v)             : amp
        dge/dt = -ge / (1.0 * ms)                       : 1
        dgi/dt = -gi / (2.0 * ms)                       : 1
        dtheta/dt = -theta / (tc_theta)                 : volt
        '''
eqs_stdp_ee = '''
        post2before                             : 1
        dpre   / dt = -pre   / (tc_pre_ee)      : 1 (clock-driven)
        dpost1 / dt = -post1 / (tc_post_1_ee)   : 1 (clock-driven)
        dpost2 / dt = -post2 / (tc_post_2_ee)   : 1 (clock-driven)
        w                                       : 1
        '''
eqs_stdp_pre_ee  = 'pre = 1.; w -= nu_ee_pre0 * post1; ge_post += w'
eqs_stdp_post_ee = 'post2before = post2; w += nu_ee_post0 * pre * post2before; post1 = 1.; post2 = 1.'

b.ion()
neuron_groups  = {}
input_groups   = {}
synapses       = {}
stdp_methods   = {}
rate_monitors  = {}
spike_monitors = {}
spike_counters = {}
population_names = ['A']

neuron_groups_e = NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e, method = 'euler')
neuron_groups['e'] = neuron_groups_e

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------
for name in population_names:
    print 'create neuron group', name
    neuron_groups[name+'e'] = neuron_groups['e']
    neuron_groups[name+'e'].v = v_rest_e - 40. * mV
    neuron_groups['e'].theta = np.ones((n_e)) * 20.0 * mV
    
    print 'create monitors for', name
    #rate_monitors[name+'e'] = PopulationRateMonitor(neuron_groups[name+'e'])
    #spike_counters[name+'e'] = SpikeMonitor(neuron_groups[name+'e'], record=False)
    #spike_monitors[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])

#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------  
input_population_names = ['X']
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = PoissonGroup(n_input, 0* Hz)
    #rate_monitors[name+'e'] = PopulationRateMonitor(input_groups[name+'e'])

data_path = './'
weight_path = data_path + 'random/'  
input_connection_names = ['XA']
input_conn_names = ['ee_input'] 
for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + '.npy')
        print weightMatrix.shape
        synapses_XeAe = Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]], model = eqs_stdp_ee, on_pre = eqs_stdp_pre_ee, on_post = eqs_stdp_post_ee, method = 'linear')
        synapses[connName] = synapses_XeAe
        synapses[connName].connect()
        synapses[connName].delay = rand() * delay['ee_input'];
        synapses[connName].w = weightMatrix.flatten()

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
fig_num = 1 
input_weight_monitor, fig_weights = plot_2d_input_weights()
#fig_num += 1

input_groups['Xe'].rates = 0 * Hz

defaultclock.dt = 0.5 * ms;

run(0 * second)
j = 0

while j < 5:
    normalize_weights()
    input_groups['Xe'].rates = training['x'][0, :, :].reshape((n_input)) / 8. * 2. * Hz
    run(single_example_time)

    update_2d_input_weights(input_weight_monitor, fig_weights)
    input_groups['Xe'].rates = 0 * Hz
    run(resting_time)
    j += 1


b.ioff()
b.show()        
    
    
    
    
    