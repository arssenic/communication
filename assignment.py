import numpy as np
from scipy import signal
from commpy.filters import rrcosfilter

class DigitalCommunicationSystem:
    def __init__(self, n_symbols=1000, sps=16, snr_db=10):
        self.n_symbols = n_symbols
        self.sps = sps  # samples per symbol
        self.snr_db = snr_db
        
    def generate_binary_stream(self):
        """Generate random binary stream for 32-QAM (5 bits per symbol)"""
        bits_per_symbol = 5  # 32-QAM uses 5 bits per symbol
        total_bits = self.n_symbols * bits_per_symbol
        x=np.random.randint(0, 2, total_bits)
        print(x.shape)
        return x
    
    def modulate_32qam(self, bit_stream):
        """32-QAM modulation mapper"""
        # Ensure the bit stream length is divisible by 5
        bits_per_symbol = 5
        pad_length = (bits_per_symbol - (len(bit_stream) % bits_per_symbol)) % bits_per_symbol
        if pad_length:
            bit_stream = np.pad(bit_stream, (0, pad_length), 'constant')
        
        # Reshape bit stream into groups of 5 bits
        bit_groups = bit_stream.reshape(-1, bits_per_symbol)
        
        # Initialize complex symbols array
        symbols = np.zeros(len(bit_groups), dtype=complex)
        
        for i, bits in enumerate(bit_groups):
            # Convert 5 bits to decimal using binary weights
            decimal_value = np.sum(bits * np.array([16, 8, 4, 2, 1]))
            
            # Map to constellation points
            # This is a simplified 32-QAM constellation mapping
            row = decimal_value // 6
            col = decimal_value % 6
            
            # Generate complex symbol
            real_part = 2 * row - 2.5
            imag_part = 2 * col - 2.5
            symbols[i] = (real_part + 1j * imag_part) / np.sqrt(10)  # Normalize energy
            
        return symbols
    
    def pulse_shape(self, symbols):
        """Apply pulse shaping filter"""
        # RRC filter parameters
        beta = 0.35  # roll-off factor
        t_symbol = 1.0
        
        # Generate RRC filter
        t = np.arange(-4, 4, 1/self.sps)
        h_rrc = rrcosfilter(len(t), beta, t_symbol, self.sps)[1]
        
        # Upsample symbols
        symbols_up = np.zeros(len(symbols) * self.sps, dtype=complex)
        symbols_up[::self.sps] = symbols
        
        # Apply filter
        filtered = signal.convolve(symbols_up, h_rrc, mode='same')
        return filtered
    
    def baseband_to_passband(self, baseband_signal, carrier_freq):
        """Convert baseband signal to passband"""
        t = np.arange(len(baseband_signal)) / self.sps
        passband_signal = np.real(baseband_signal * np.exp(1j * 2 * np.pi * carrier_freq * t))
        return passband_signal
    
    def add_awgn(self, signal):
        """Add AWGN noise according to specified SNR"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.snr_db/10))
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                        1j * np.random.randn(len(signal)))
        return signal + noise
    
    def passband_to_baseband(self, passband_signal, carrier_freq):
        """Convert passband signal to baseband"""
        # Create time vector
        t = np.linspace(0, len(passband_signal)/self.sps, len(passband_signal), endpoint=False)
        
        # Coherent downconversion
        baseband_signal = (
            passband_signal * 
            np.exp(-1j * 2 * np.pi * carrier_freq * t)
        )
        
        # Low-pass filtering
        # Ensure valid cutoff frequency
        sampling_rate = self.sps  # samples per second
        nyquist_freq = 0.5 * sampling_rate
        
        # Set cutoff to a valid frequency (less than Nyquist frequency)
        cutoff_freq = min(carrier_freq, nyquist_freq * 0.9)
        
        # Design Butterworth low-pass filter
        b, a = signal.butter(
            N=10,  # Filter order
            Wn=cutoff_freq / (0.5 * sampling_rate),  # Normalized cutoff frequency
            btype='low'
        )
        
        # Apply low-pass filter
        baseband_signal_filtered = signal.filtfilt(b, a, baseband_signal)
        
        return baseband_signal_filtered

    def receive(self, received_signal):
        """Implement receiver operations"""
        # Matched filtering (same as pulse shaping filter)
        beta = 0.35
        t = np.arange(-4, 4, 1/self.sps)
        h_rrc = rrcosfilter(len(t), beta, 1.0, self.sps)[1]
        matched_filtered = signal.convolve(received_signal, h_rrc, mode='same')
        
        # Sample at symbol points
        sampled = matched_filtered[4*self.sps::self.sps]  # Skip initial delay
        
        # Demodulate
        demod_bits = self.demodulate_32qam(sampled)
        
        return demod_bits
    
    def demodulate_32qam(self, received_symbols):
        """32-QAM demodulation"""
        # Scale received symbols
        scaled_symbols = received_symbols * np.sqrt(10)
        
        # Make decisions based on constellation regions
        real_part = np.real(scaled_symbols)
        imag_part = np.imag(scaled_symbols)
        
        # Initialize bit array
        bits = np.zeros(len(received_symbols) * 5, dtype=int)
        
        for i, (real, imag) in enumerate(zip(real_part, imag_part)):
            # Quantize to closest constellation point
            row = np.round((real + 2.5) / 2)
            col = np.round((imag + 2.5) / 2)
            
            # Clip to valid range
            row = np.clip(row, 0, 5)
            col = np.clip(col, 0, 5)
            
            # Convert to decimal value
            decimal = int(row * 6 + col)
            
            # Convert to bits using binary weights
            bit_values = [(decimal >> j) & 1 for j in range(4, -1, -1)]
            bits[i*5:(i+1)*5] = bit_values
            
        return bits
    
    def calculate_ber(self, original_bits, received_bits):
        """Calculate Bit Error Rate"""
        errors = np.sum(original_bits != received_bits)
        ber = errors / len(original_bits)
        return ber
    
    def simulate(self):
        """Run complete simulation"""
        # Transmitter
        # Transmitter parameters
        carrier_freq = 2e6  # 2 MHz carrier frequency
        bits = self.generate_binary_stream()
        symbols = self.modulate_32qam(bits)
        shaped_signal = self.pulse_shape(symbols)
        
        # Convert to passband
        passband_signal = self.baseband_to_passband(shaped_signal, carrier_freq)

        # Channel
        received_signal = self.add_awgn(passband_signal)

        # Convert back to baseband
        baseband_signal = self.passband_to_baseband(received_signal, carrier_freq)
        
        # Receiver
        received_bits = self.receive(baseband_signal)
        
        # Trim bits to match (due to filter delays)
        min_len = min(len(bits), len(received_bits))
        ber = self.calculate_ber(bits[:min_len], received_bits[:min_len])
        
        return ber

# Example usage
if __name__ == "__main__":
    # Test system with different SNR values
    snr_values = np.arange(-5, 21, 5)
    ber_values = []
    
    for snr in snr_values:
        system = DigitalCommunicationSystem(n_symbols=10000, sps=16, snr_db=snr)
        ber = system.simulate()
        ber_values.append(ber)
        print(f"SNR: {snr} dB, BER: {ber:.6f}")
    
    