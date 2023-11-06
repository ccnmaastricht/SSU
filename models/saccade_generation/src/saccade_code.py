"""
This module is an adapted version of the spiking neural network model of the saccade 
generator in the reticular formation, originally developed by Anno Kurthwith support from Sacha van Albada. 
The original work can be found at https://github.com/ccnmaastricht/spiking_saccade_generator.

Adaptations were made to integrate with the larger project structure and to meet specific 
requirements of our implementation. We thank Anno Kurth for their contribution to this project.

:license: CC BY-NC-SA 4.0, see CC_BY_NC_SA_LICENSE.md for more details.
"""


import json
import nest
import numpy as np

class SaccadeGenerator():
    def __init__(self):
        '''
        Initialize the SaccadeGenerator.
        '''
        
        with open('parameters/population_parameters.json') as f:
            self.population_parameters = json.load(f)

        with open('parameters/connection_parameters.json') as f:
            self.connection_parameters = json.load(f)

        with open('parameters/simulation_parameters.json') as f:
            self.simulation_parameters = json.load(f)


        self.simulation_time = self.simulation_parameters['simulation_time']

        self.eye_position = np.zeros(2)
        

    def constructor(self):
        '''
        Construct spiking neural network model of saccade generator
        '''
        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        saccade_generator = self.construct_saccade_generator()
        
        # fetch horizontal and vertical saccade generators
        horizontal_sg = saccade_generator['horizontal']
        vertical_sg = saccade_generator['vertical']

        # input populations
        self.left_llbn = horizontal_sg['LLBN_l']
        self.right_llbn = horizontal_sg['LLBN_r']
        self.up_llbn = vertical_sg['LLBN_u']
        self.down_llbn = vertical_sg['LLBN_d']

        # output populations
        self.left_ebn = horizontal_sg['EBN_l']
        self.right_ebn = horizontal_sg['EBN_r']
        self.up_ebn = vertical_sg['EBN_u']
        self.down_ebn = vertical_sg['EBN_d']

        # construct input and output spike detectors
        self.construct_dc_generators()
        self.construct_recording_devices()

        
    def reset(self):
        '''
        Reset the SaccadeGenerator.
        '''
        self.eye_position = np.zeros(2) # reset eye position


    def simulate(self, amplitude):
        '''
        Simulate the SaccadeGenerator.

        Args:
            amplitude (tuple): The amplitude of the saccade
        '''
        amplitude_left, amplitude_right, amplitude_up, amplitude_down = amplitude

        self.constructor()

        nest.SetStatus(self.dc_generator_left, {'amplitude' : amplitude_left})
        nest.SetStatus(self.dc_generator_right, {'amplitude' : amplitude_right})
        nest.SetStatus(self.dc_generator_up, {'amplitude' : amplitude_up})
        nest.SetStatus(self.dc_generator_down, {'amplitude' : amplitude_down})

        nest.Simulate(self.simulation_time)

        # fetch spike times of EBNs
        spike_times_left = nest.GetStatus(self.spike_detector_left, 'events')[0]['times']
        spike_times_right = nest.GetStatus(self.spike_detector_right, 'events')[0]['times']
        spike_times_up = nest.GetStatus(self.spike_detector_up, 'events')[0]['times']
        spike_times_down = nest.GetStatus(self.spike_detector_down, 'events')[0]['times']

        # compute spike rates of EBNs
        rate_left = len(spike_times_left) / self.simulation_time
        rate_right = len(spike_times_right) / self.simulation_time
        rate_up = len(spike_times_up) / self.simulation_time
        rate_down = len(spike_times_down) / self.simulation_time

        # compute eye position
        horizontal_displacement = (rate_right - rate_left) / 22.38 * 180.0
        vertical_displacement = (rate_up - rate_down) / 22.38 * 90.0

        self.eye_position[0] += horizontal_displacement
        self.eye_position[1] += vertical_displacement
        

    def construct_dc_generators(self):
        '''
        Construct dc generators for LLBNs
        '''
        self.dc_generator_left = nest.Create('dc_generator', 1)
        self.dc_generator_right = nest.Create('dc_generator', 1)
        self.dc_generator_up = nest.Create('dc_generator', 1)
        self.dc_generator_down = nest.Create('dc_generator', 1)

        nest.SetStatus(self.dc_generator_left, {'amplitude' : 0.,
                                            'start' : 0.,
                                            'stop' : self.simulation_time})

        nest.SetStatus(self.dc_generator_right, {'amplitude' : 0.,
                                            'start' : 0.,
                                            'stop' : self.simulation_time})

        nest.SetStatus(self.dc_generator_up, {'amplitude' : 0.,
                                            'start' : 0.,
                                            'stop' : self.simulation_time})
        
        nest.SetStatus(self.dc_generator_down, {'amplitude' : 0.,
                                            'start' : 0.,
                                            'stop' : self.simulation_time})
        
        # connect dc generators to LLBNs
        nest.Connect(self.dc_generator_left, self.left_llbn)
        nest.Connect(self.dc_generator_right, self.right_llbn)
        nest.Connect(self.dc_generator_up, self.up_llbn)
        nest.Connect(self.dc_generator_down, self.down_llbn)

    
    def construct_recording_devices(self):
        '''
        Construct spike detectors for EBNs
        '''
        self.spike_detector_right = nest.Create('spike_detector', 1)
        self.spike_detector_left = nest.Create('spike_detector', 1)
        self.spike_detector_up = nest.Create('spike_detector', 1)
        self.spike_detector_down = nest.Create('spike_detector', 1)

        # connect devices
        nest.Connect(self.left_ebn, self.spike_detector_left)
        nest.Connect(self.right_ebn, self.spike_detector_right)
        nest.Connect(self.up_ebn, self.spike_detector_up)
        nest.Connect(self.down_ebn, self.spike_detector_down)


    def construct_saccade_generator(self):
        '''
        Construct model network of saccade generator for control of two
        extraocular muscles

        Returns
        -------
        saccade_generator : dict
            dictionary containing two single side saccade generators

        '''

        '''
        saccade generator for horizonal eye movement
        '''
        # saccade generator left saccades
        LLBN_l, EBN_l, IBN_l, OPN_h = self.saccade_generator_single_side()
        # saccade generator right saccades
        LLBN_r, EBN_r, IBN_r, OPN_h = self.saccade_generator_single_side(OPN_h)

        '''
        saccade generator for vertical eye movement
        '''
        # saccade generator upward saccades
        LLBN_u, EBN_u, IBN_u, OPN_v = self.saccade_generator_single_side()
        # saccade generator downward saccades
        LLBN_d, EBN_d, IBN_d, OPN_v = self.saccade_generator_single_side(OPN_v)

        horizontal_saccade_generator = {'LLBN_l' : LLBN_l,
                                        'EBN_l' : EBN_l,
                                        'IBN_l' : IBN_l,
                                        'LLBN_r' : LLBN_r,
                                        'EBN_r' : EBN_r,
                                        'IBN_r' : IBN_r,
                                        'OPN_h' : OPN_h}

        vertical_saccade_generator = {'LLBN_u' : LLBN_u,
                                    'EBN_u' : EBN_u,
                                    'IBN_u' : IBN_u,
                                    'LLBN_d' : LLBN_d,
                                    'EBN_d' : EBN_d,
                                    'IBN_d' : IBN_d,
                                    'OPN_v' : OPN_v}

        saccade_generator = {'horizontal' : horizontal_saccade_generator,
                            'vertical' : vertical_saccade_generator}

        return saccade_generator
    
    def saccade_generator_single_side(self, OPN = None):
        '''
        Construct model network of saccade generator for control of a single
        extraocular muscle

        Parameters
        ----------
        OPN : tuple
            if not NONE passed population is used as OPN population

        Returns
        -------

        LLBN : tuple
            gids of neurons in population LLBN

        EBN : tuple
            gids of neurons in population EBN

        IBN : tuple
            gids of neurons in population IBN

        OPN : tuple
            gids of neurons in population OPN
        '''

        LLBN_parameters = self.population_parameters['LLBN_parameters']
        EBN_parameters = self.population_parameters['EBN_parameters']
        IBN_parameters = self.population_parameters['IBN_parameters']
        OPN_parameters = self.population_parameters['OPN_parameters']

        connection_llbn_slbn = self.connection_parameters['connection_llbn_slbn']
        connection_llbn_opn = self.connection_parameters['connection_llbn_opn']
        connection_opn_slbn = self.connection_parameters['connection_opn_slbn']
        connection_ebn_ibn = self.connection_parameters['connection_ebn_ibn']
        connection_ibn_llbn = self.connection_parameters['connection_ibn_llbn']


        # Create populations constituing saccade generator
        LLBN_e, LLBN_i = self.create_population(**LLBN_parameters)
        LLBN = LLBN_e + LLBN_i

        EBN, SLBN_i = self.create_population(**EBN_parameters)
        SLBN = EBN + SLBN_i

        __, IBN = self.create_population(**IBN_parameters)

        if OPN == None:
            OPN_e, OPN_i = self.create_population(**OPN_parameters)
            OPN = OPN_e + OPN_i

        else:
            OPN_e = OPN[:OPN_parameters['n_ex']]
            OPN_i = OPN[OPN_parameters['n_ex']:]

        # Connect respective populations to recurrent neural network
        nest.Connect(LLBN_e, SLBN, connection_llbn_slbn['conn_spec'],
                    connection_llbn_slbn['syn_spec'])
        nest.Connect(LLBN_i, OPN, connection_llbn_opn['conn_spec'],
                    connection_llbn_opn['syn_spec'])
        nest.Connect(OPN, SLBN, connection_opn_slbn['conn_spec'],
                    connection_opn_slbn['syn_spec'])
        nest.Connect(EBN, IBN, connection_ebn_ibn['conn_spec'],
                    connection_ebn_ibn['syn_spec'])
        nest.Connect(IBN, LLBN, connection_ibn_llbn['conn_spec'],
                    connection_ibn_llbn['syn_spec'])

        return LLBN, EBN, IBN, OPN
    
    
    def create_population(self, n_ex, n_in, neuron_model, single_neuron_params,
                      noise, connection_params):
        '''
        Parameters
        ----------
        n_ex : int
            size of populations of excitatory neurons

        n_in : int
            size of population of inhibitory neurons

        neuron_model : string
            neuron model in nest used in network

        single_neuron_params: dict
            parameter of single neurons in network

        noise : float
            provided Gaussian noise to network

        connetion_params : dict
            parameters describing connectivity of network

        Returns
        ------
        neurons_ex : tuple
            nest-gids of neurons in excitatory population

        neurons_in : tuple
            nest-gids of neurons in inhibitory population
        '''

        assert type(n_ex)==int , 'Number of excitatory neurons must be int'
        assert type(n_in)==int , 'Number of excitatory neurons must be int'
        assert type(noise)==float, 'Value of noise must be float'

        # Create neuron population and noise generator
        if n_ex > 0:
            neurons_ex = nest.Create(neuron_model, n_ex, single_neuron_params)
        else :
            neurons_ex = ()

        if n_in > 0:
            neurons_in = nest.Create(neuron_model, n_in, single_neuron_params)
        else :
            neurons_in = ()

        if noise > 0:
            noise = nest.Create('noise_generator', 1, {'std' : noise})

        neurons = neurons_ex + neurons_in

        connectivity_ex = connection_params['ex']
        connectivity_in = connection_params['in']

        # Connect neurons and noise
        nest.Connect(neurons_ex, neurons, connectivity_ex['conn_spec'],
                    connectivity_ex['syn_spec'])
        nest.Connect(neurons_in, neurons, connectivity_in['conn_spec'],
                    connectivity_in['syn_spec'])

        nest.Connect(noise, neurons)

        return neurons_ex, neurons_in

