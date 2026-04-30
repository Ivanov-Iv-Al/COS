import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import convolution_matrix

h = np.array([0.19 + 0.56j, 0.45 - 1.28j, -0.14 - 0.53j])
snr_14 = 14
alphabet = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

n_symbols = 1000
tx = np.random.choice(alphabet, n_symbols)

def add_noise(s, snr_db):
    p = np.mean(np.abs(s) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise = np.sqrt(p / (2 * snr_lin)) * (np.random.randn(len(s)) + 1j * np.random.randn(len(s)))
    return s + noise


rx_raw = np.convolve(tx, h, mode='full')[:n_symbols]
rx_noisy = add_noise(rx_raw, snr_14)

Lf = 6

H_mat = convolution_matrix(h, Lf, mode='full')

best_d = 0
min_error = float('inf')
best_f = None

for d in range(H_mat.shape[0]):

    ed = np.zeros(H_mat.shape[0])
    ed[d] = 1

    f, residuals, rank, s = np.linalg.lstsq(H_mat, ed, rcond=None)

    error = np.linalg.norm(H_mat @ f - ed) ** 2

    if error < min_error:
        min_error = error
        best_d = d
        best_f = f

rx_equalized = np.convolve(rx_noisy, best_f, mode='same')

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(rx_noisy.real, rx_noisy.imag, s=5, alpha=0.6, color='red')
plt.title(f"До эквалайзера (SNR 14 dB)\nМСИ размывает созвездие")
plt.grid(True)

plt.subplot(122)
plt.scatter(rx_equalized.real, rx_equalized.imag, s=5, alpha=0.6, color='green')
plt.title(f"После LS эквалайзера (d={best_d})\nМСИ скомпенсирована")
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Импульсная характеристика канала h: {h}")
print(f"Матрица свертки H (первые 3 строки):\n{H_mat[:3, :]}")
print(f"Оптимальная задержка d: {best_d}")
print(f"Коэффициенты эквалайзера f_d: {best_f}")
print(f"Минимальная ошибка J(d): {min_error:.6f}")
