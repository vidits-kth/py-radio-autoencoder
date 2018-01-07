# COPYRIGHT_NOTICE

import numpy as np

from py_itpp.base import bvec, cvec, ivec
from py_itpp.base import random

from py_itpp.comm.modulator import modulator_2d, soft_method
from py_itpp.comm.error_counters import BLERC
from py_itpp.comm.hammcode import hamming_code

def block_error_ratio_hamming_awgn(snr_db, block_size):
    
    mapping_k_m = {4: 3} # Mapping from k (block size) to m. m = 3 implies (7,4) code
    m = mapping_k_m[block_size]
     
    '''Hamming encoder and decoder instance'''
    hamm = hamming_code(m)
    n = pow(2,m) - 1 # channel use
    rate = float(block_size)/float(n)
    
    '''Generate random bits'''
    nrof_bits = 10000 * block_size
    source_bits = random.randb(nrof_bits)
    
    '''Encode the bits'''
    encoded_bits = hamm.encode(source_bits)
    
    '''Modulate the bits'''
    modulator_ = modulator_2d()
    constellation = cvec('-1+0i, 1+0i')
    symbols = ivec('0, 1')
    modulator_.set(constellation, symbols)
    tx_signal = modulator_.modulate_bits(encoded_bits)
    
    '''Add the effect of channel to the signal'''
    noise_variance = 1.0 / (rate * pow(10, 0.1 * snr_db))
    noise = random.randn_c(tx_signal.length())
    noise *= np.sqrt(noise_variance)
    rx_signal = tx_signal + noise
    
    '''Demodulate the signal'''
    demodulated_bits = modulator_.demodulate_bits(rx_signal)
    
    '''Decode the received bits'''
    decoded_bits = hamm.decode(demodulated_bits) 
    
    '''Calculate the block error ratio'''
    blerc = BLERC(block_size)
    blerc.count(source_bits, decoded_bits)
    return blerc.get_errorrate()