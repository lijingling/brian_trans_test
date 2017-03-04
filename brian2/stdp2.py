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
    
    connMatrix = connections[name].w
    weight_matrix = np.copy(connMatrix)
    #print weight_matrix.shape
    if n_e > 1 :
        rearranged_weights = np.zeros((num_values_col, num_values_row))
        for i in xrange(n_e_sqrt):
            for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    else:
        temp = (n_in_sqrt, n_in_sqrt)
        rearranged_weights = np.zeros(temp)
        rearranged_weights = weight_matrix.reshape(n_in_sqrt, n_in_sqrt)
    #print rearranged_weights.shape
    return rearranged_weights

def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = figure(fig_num, figsize = (18, 18))
    im2 = imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_ee, cmap = cmap.get_cmap('hot_r'))
    colorbar(im2)
    title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

def plot_performance(fig_num):
    num_evaluations = int(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)
    fig = figure(fig_num, figsize = (5, 5))
    fig_num += 1
    ax = fig.add_subplot(111)
    im2, = ax.plot(time_steps, performance) #my_cmap
    ylim(ymax = 100)
    title('Classification performance')
    fig.canvas.draw()
    return im2, performance, fig_num, fig

def normalize_weights():
    for connName in connections:
        if connName[1] == 'e' and connName[3] == 'e':
            connection = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            print colSums
            colFactors = weight['ee_input']/colSums
            if n_e == 1:
                connections[connName].w *= colFactors
            else:
                for j in xrange(n_e):#
                    connections[connName].w[:,j] *= colFactors[j]

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

def save_connections(ending = ''):
    print 'save connections'
    for connName in save_conns:
        connMatrix = connections[connName][:]
#         connListSparse = ([(i,j[0],j[1]) for i in xrange(connMatrix.shape[0]) for j in zip(connMatrix.rowj[i],connMatrix.rowdata[i])])
        connListSparse = ([(i,j,connMatrix[i,j]) for i in xrange(connMatrix.shape[0]) for j in xrange(connMatrix.shape[1]) ])
        np.save(data_path + 'weights/' + connName + ending, connListSparse)

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

def get_current_performance(performance, current_example_num):
    current_evaluation = int(current_example_num/update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num
    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])
    performance[current_evaluation] = correct / float(update_interval) * 100
    return performance

def update_performance_plot(im, performance, current_example_num, fig):
    performance = get_current_performance(performance, current_example_num)
    im.set_ydata(performance)
    fig.canvas.draw()
    return im, performance

#------------------------------------------------------------------------------ 
# load MNIST
#------------------------------------------------------------------------------
start = time.time()
training = get_labeled_data(MNIST_data_path + 'training')
end = time.time()
print 'time needed to load training set:', end - start
 
start = time.time()
testing = get_labeled_data(MNIST_data_path + 'testing', bTrain = False)
end = time.time()
print 'time needed to load test set:', end - start

#------------------------------------------------------------------------------ 
# set parameters and equations
#------------------------------------------------------------------------------
test_mode = False

np.random.seed(0)
data_path = './'
if test_mode:
    weight_path = data_path + 'weights/'
    num_examples = 10000 * 1
    use_testing_set = True
    do_plot_performance = False
    record_spikes = True
    ee_STDP_on = False
    update_interval = num_examples
else:
    weight_path = data_path + 'random/'  
    num_examples = 60000 * 3
    use_testing_set = False
    do_plot_performance = True
    if num_examples <= 60000:    
        record_spikes = True
    else:
        record_spikes = True
    ee_STDP_on = True

ending = ''
n_input = 784
n_e = 1

single_example_time =   0.35 * second
resting_time = 0.15 * second
runtime = num_examples * (single_example_time + resting_time)

if num_examples <= 10000:    
    update_interval = num_examples
    weight_update_interval = 20
else:
    update_interval = 10000
    weight_update_interval = 100
    
if num_examples <= 60000:    
    save_connections_interval = 10000
else:
    save_connections_interval = 10000
    update_interval = 10000

v_rest_e = -65. * mV 
v_rest_i = -60. * mV 
v_reset_e = -65. * mV
v_reset_i = -45. * mV
v_thresh_e = -52. * mV
v_thresh_i = -40. * mV
refrac_e = 5. * ms
refrac_i = 2. * ms

weight = {}
save_conns = ['XeAe']
weight['ee_input'] = 78.
wmax_ee = 1.0


###function of e neuron
if test_mode:
    scr_e = 'v = v_reset_e; timer = 0*ms'
else:
    tc_theta = 1e7 * ms
    theta_plus_e = 0.05 * mV
    scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
offset = 20.0*mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

neuron_eqs_e = '''
        dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (100*ms)  : volt
        I_synE = ge * nS *         -v                           : amp
        I_synI = gi * nS * (-100.*mV-v)                          : amp
        dge/dt = -ge/(1.0*ms)                                   : 1
        dgi/dt = -gi/(2.0*ms)                                  : 1
        '''
if test_mode:
    neuron_eqs_e += '\n  theta      :volt'
else:
    neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
#neuron_eqs_e += '\n  dtimer/dt = 100.0  : ms'

neuron_groups = {}
population_names = ['A']

neuron_groups['e'] = NeuronGroup(n_e*len(population_names), neuron_eqs_e, threshold= v_thresh_e, refractory= refrac_e, reset= scr_e)

#------------------------------------------------------------------------------ 
# create network population and recurrent connections
#------------------------------------------------------------------------------
rate_monitors = {}
spike_counters = {}
spike_monitors = {}
populationnum = 0
for name in population_names:
    print 'create neuron group', name
    neuron_groups[name+'e'] = neuron_groups['e'][n_e*populationnum:n_e*(populationnum+1)]
    neuron_groups[name+'e'].v = v_rest_e - 40. * mV
    neuron_groups['e'].theta = np.ones((n_e)) * 20.0*mV
    

    print 'create monitors for', name
    rate_monitors[name+'e'] = PopulationRateMonitor(neuron_groups[name+'e'])
    spike_counters[name+'e'] = SpikeMonitor(neuron_groups[name+'e'], record=False)
    
    if record_spikes:
        spike_monitors[name+'e'] = SpikeMonitor(neuron_groups[name+'e'])
    populationnum += 1
    
fig_num = 1 
result_monitor = np.zeros((update_interval,n_e))
if record_spikes:
    figure(fig_num)
    fig_num += 1
    ion()
    brian_plot(spike_monitors['Ae'])
     
#------------------------------------------------------------------------------ 
# create input population and connections from input populations 
#------------------------------------------------------------------------------  
pop_values = [0,0,0]
input_population_names = ['X']
input_groups = {}
rate_monitors = {}
for i,name in enumerate(input_population_names):
    input_groups[name+'e'] = PoissonGroup(n_input, 0* Hz)
    rate_monitors[name+'e'] = PopulationRateMonitor(input_groups[name+'e'])

input_connection_names = ['XA']
input_conn_names = ['ee_input'] 
connections = {}
delay = {}
delay['ee_input'] = 'rand()*10*ms'
delay['ei_input'] = 'rand()*5*ms'
for name in input_connection_names:
    print 'create connections between', name[0], 'and', name[1]
    for connType in input_conn_names:
        connName = name[0] + connType[0] + name[1] + connType[1]
        weightMatrix = get_matrix_from_file(weight_path + connName + ending + '.npy')
        print weightMatrix.shape
        connections[connName] = Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]], 'w : siemens', on_pre='ge += w')
        connections[connName].connect()
        connections[connName].delay = delay[connType]
        connections[connName].w = weightMatrix.flatten() * siemens

#------------------------------------------------------------------------------ 
# run the simulation and set inputs
#------------------------------------------------------------------------------ 
previous_spike_count = np.zeros(n_e)
assignments = np.zeros(n_e)
input_numbers = [0] * num_examples
outputNumbers = np.zeros((num_examples, 10))

if not test_mode:
    input_weight_monitor, fig_weights = plot_2d_input_weights()
    fig_num += 1

if do_plot_performance:
    performance_monitor, performance, fig_num, fig_performance = plot_performance(fig_num)

for i,name in enumerate(input_population_names):
    input_groups[name+'e'].rates = 0

j = 0
input_intensity = 2.
start_input_intensity = input_intensity

while j < (int(num_examples)):
    if test_mode:
        if use_testing_set:
            rates = testing['x'][j%10000,:,:].reshape((n_input)) / 8. *  input_intensity
        else:
            #rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
            rates = training['x'][0,:,:].reshape((n_input)) / 8. *  input_intensity
    else:
        normalize_weights()
        rates = training['x'][j%60000,:,:].reshape((n_input)) / 8. *  input_intensity
    input_groups['Xe'].rates = rates * Hz
#     print 'run number:', j+1, 'of', int(num_examples)
    run(single_example_time, report='text')    
    
    if j % update_interval == 0 and j > 0:
        assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j])
    if j % weight_update_interval == 0 and not test_mode:
        update_2d_input_weights(input_weight_monitor, fig_weights)
    if j % save_connections_interval == 0 and j > 0 and not test_mode:
        save_connections(str(j))
        save_theta(str(j))
    
    current_spike_count = np.asarray(spike_counters['Ae'].count[:]) - previous_spike_count
    previous_spike_count = np.copy(spike_counters['Ae'].count[:])
    if np.sum(current_spike_count) < 5:
        input_intensity += 1
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rates = 0
        run(resting_time)
    else:
        result_monitor[j%update_interval,:] = current_spike_count
        if test_mode and use_testing_set:
            input_numbers[j] = testing['y'][j%10000][0]
        else:
            input_numbers[j] = training['y'][j%60000][0]
        outputNumbers[j,:] = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:])
        if j % 100 == 0 and j > 0:
            print 'runs done:', j, 'of', int(num_examples)
        if j % update_interval == 0 and j > 0:
            if do_plot_performance:
                unused, performance = update_performance_plot(performance_monitor, performance, j, fig_performance)
                print 'Classification performance', performance[:(j/float(update_interval))+1]
        for i,name in enumerate(input_population_names):
            input_groups[name+'e'].rate = 0
        run(resting_time)
        input_intensity = start_input_intensity
        j += 1
    
#------------------------------------------------------------------------------ 
# save results
#------------------------------------------------------------------------------ 
print 'save results'
if not test_mode:
    save_theta()
if not test_mode:
    save_connections()
else:
    np.save(data_path + 'activity/resultPopVecs' + str(num_examples), result_monitor)
    np.save(data_path + 'activity/inputNumbers' + str(num_examples), input_numbers)

#------------------------------------------------------------------------------ 
# plot results
#------------------------------------------------------------------------------ 
if rate_monitors:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(rate_monitors):
        subplot(len(rate_monitors), 1, i)
        plot(rate_monitors[name].times/b.second, rate_monitors[name].rate, '.')
        title('Rates of population ' + name)
    
if spike_monitors:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_monitors):
        subplot(len(spike_monitors), 1, i)
        raster_plot(spike_monitors[name])
        title('Spikes of population ' + name)
        
if spike_counters:
    figure(fig_num)
    fig_num += 1
    for i, name in enumerate(spike_counters):
        subplot(len(spike_counters), 1, i)
        plot(spike_counters['Ae'].count[:])
        title('Spike count of population ' + name)

plot_2d_input_weights()
ioff()
show()        
    
    
    
    
    