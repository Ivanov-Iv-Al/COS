import numpy as np
import matplotlib.pyplot as plt

alphabet = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
n_symbols = 1000
tx_symbols = np.random.choice(alphabet, n_symbols)

def add_awgn(signal, snr_db):
    sig_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

rx_noisy = add_awgn(tx_symbols, 8)

h = np.array([0.19 + 0.56j, 0.45 - 1.28j, -0.14 - 0.53j])

rx_1tap = np.convolve(tx_symbols, [h[0]], mode='same')

rx_3tap = np.convolve(tx_symbols, h, mode='same')


tr = np.array([0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0])
tr_center = tr[5:21]

m = 1 - 2 * tr_center
akf = np.correlate(m, m, mode='full')

L = len(h)
N = len(m)

T = np.zeros((N - L + 1, L), dtype=complex)
for i in range(N - L + 1):
    T[i, :] = m[i + L - 1 : i - 1 if i > 0 else None : -1]

y_clean = np.convolve(m, h, mode='valid')
y_noisy = add_awgn(y_clean, 14)

h_est = np.linalg.lstsq(T, y_noisy, rcond=None)[0]
mse = np.mean(np.abs(h - h_est)**2)

plt.figure(figsize=(12, 10))

plt.subplot(321)
plt.scatter(rx_noisy.real, rx_noisy.imag, s=1)
plt.title("2. QPSK + AWGN (SNR 8dB)")

plt.subplot(322)
plt.stem(np.abs(h))
plt.title("3. Модуль ИХ канала")

plt.subplot(323)
plt.scatter(rx_1tap.real, rx_1tap.imag, s=1)
plt.title("4. Канал (1 отсчет)")

plt.subplot(324)
plt.scatter(rx_3tap.real, rx_3tap.imag, s=1)
plt.title("5. Канал (3 отсчета - МСИ)")

plt.subplot(325)
plt.plot(np.abs(akf))
plt.title("6. АКФ обучающей последовательности")

plt.subplot(326)
plt.stem(np.abs(h), linefmt='b-', markerfmt='bo', label='True')
plt.stem(np.abs(h_est), linefmt='r--', markerfmt='rx', label='LS Est')
plt.title(f"9. Сравнение ИХ (MSE: {mse:.5f})")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Заданная ИХ: {h}")
print(f"Оценка ИХ:   {h_est}")
print(f"MSE: {mse}")
