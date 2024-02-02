import numpy as np
import matplotlib.pyplot as plt

def dF1(t, s, e, i, r, beta):
    return -beta * i * s

def dF2(t, s, e, i, r, beta, sigma):
    return beta * i * s - sigma * e

def dF3(t, s, e, i, r, sigma, gamma):
    return sigma * e - gamma * i

def dF4(t, s, e, i, r, gamma):
    return gamma * i

def rk4(h, t, s, e, i, r, beta, sigma, gamma):
    F1s = h * dF1(t, s, e, i, r, beta)
    F1e = h * dF2(t, s, e, i, r, beta, sigma)
    F1i = h * dF3(t, s, e, i, r, sigma, gamma)
    F1r = h * dF4(t, s, e, i, r, gamma)

    F2s = h * dF1(t + 0.5 * h, s + 0.5 * F1s, e + 0.5 * F1e, i + 0.5 * F1i, r + 0.5 * F1r, beta)
    F2e = h * dF2(t + 0.5 * h, s + 0.5 * F1s, e + 0.5 * F1e, i + 0.5 * F1i, r + 0.5 * F1r, beta, sigma)
    F2i = h * dF3(t + 0.5 * h, s + 0.5 * F1s, e + 0.5 * F1e, i + 0.5 * F1i, r + 0.5 * F1r, sigma, gamma)
    F2r = h * dF4(t + 0.5 * h, s + 0.5 * F1s, e + 0.5 * F1e, i + 0.5 * F1i, r + 0.5 * F1r, gamma)

    F3s = h * dF1(t + 0.5 * h, s + 0.5 * F2s, e + 0.5 * F2e, i + 0.5 * F2i, r + 0.5 * F2r, beta)
    F3e = h * dF2(t + 0.5 * h, s + 0.5 * F2s, e + 0.5 * F2e, i + 0.5 * F2i, r + 0.5 * F2r, beta, sigma)
    F3i = h * dF3(t + 0.5 * h, s + 0.5 * F2s, e + 0.5 * F2e, i + 0.5 * F2i, r + 0.5 * F2r, sigma, gamma)
    F3r = h * dF4(t + 0.5 * h, s + 0.5 * F2s, e + 0.5 * F2e, i + 0.5 * F2i, r + 0.5 * F2r, gamma)

    F4s = h * dF1(t + h, s + F3s, e + F3e, i + F3i, r + F3r, beta)
    F4e = h * dF2(t + h, s + F3s, e + F3e, i + F3i, r + F3r, beta, sigma)
    F4i = h * dF3(t + h, s + F3s, e + F3e, i + F3i, r + F3r, sigma, gamma)
    F4r = h * dF4(t + h, s + F3s, e + F4e, i + F4i, r + F3r, gamma)

    s_nast = s + (1 / 6) * (F1s + 2 * F2s + 2 * F3s + F4s)
    e_nast = e + (1 / 6) * (F1e + 2 * F2e + 2 * F3e + F4e)
    i_nast = i + (1 / 6) * (F1i + 2 * F2i + 2 * F3i + F4i)
    r_nast = r + (1 / 6) * (F1r + 2 * F2r + 2 * F3r + F4r)

    return s_nast, e_nast, i_nast, r_nast

def seir(h, t_max, s0, e0, i0, r0, beta, sigma, gamma):
    kroki_czasowe = np.arange(0, t_max, h)
    wyniki_s = []
    wyniki_e = []
    wyniki_i = []
    wyniki_r = [] 
    s = s0
    e = e0
    i = i0
    r = r0

    for t in kroki_czasowe:
        wyniki_s.append(s)
        wyniki_e.append(e)
        wyniki_i.append(i)
        wyniki_r.append(r)
        s, e, i, r = rk4(h, t, s, e, i, r, beta, sigma, gamma)

    return kroki_czasowe, wyniki_s, wyniki_e, wyniki_i, wyniki_r

# Warunki początkowe
s0 = 0.99
e0 = 0.01
i0 = 0.0
r0 = 0.0

t_max = 50
h = 0.1

# Parametry modelu
beta = 1.0
sigma = 1.0
gamma = 0.1


# Zadanie 1
kroki_czasowe, wyniki_s, wyniki_e, wyniki_i, wyniki_r = seir(h, t_max, s0, e0, i0, r0, beta, sigma, gamma)

# Zadanie 2 - Zmniejszona wartość beta
beta_2 = beta / 2.0
kroki_czasowe_2, wyniki_s_b_2, wyniki_e_b_2, wyniki_i_b_2, wyniki_r_b_2 = seir(h, t_max, s0, e0, i0, r0, beta_2, sigma, gamma)

# Zadanie 3 
beta_wieksze_R0 = 2
beta_mniejsze_R0 = 0
kroki_czasowe_wieksze_R0, wyniki_s_wieksze_R0, wyniki_e_wieksze_R0, wyniki_i_wieksze_R0, wyniki_r_wieksze_R0 = seir(h, t_max, s0, e0, i0, r0, beta_wieksze_R0, sigma, gamma)
kroki_czasowe_mniejsze_R0, wyniki_s_mniejsze_R0, wyniki_e_mniejsze_R0, wyniki_i_mniejsze_R0, wyniki_r_mniejsze_R0 = seir(h, t_max, s0, e0, i0, r0, beta_mniejsze_R0, sigma, gamma)


# Wykresy
plt.figure(figsize=(14, 8))

# Wykres dla zadania 1
plt.subplot(3, 2, 1)
plt.plot(kroki_czasowe, wyniki_s, label='Susceptible (s)')
plt.plot(kroki_czasowe, wyniki_e, label='Exposed (e)')
plt.plot(kroki_czasowe, wyniki_i, label='Infectious (i)')
plt.plot(kroki_czasowe, wyniki_r, label='Removed (r)')
plt.title('SEIR Model - Metoda Rungego-Kutty czwartego rzędu (β = 1)')
plt.xlabel('Czas')
plt.legend()

# Wykres dla zadania 2
plt.subplot(3, 2, 2)
plt.plot(kroki_czasowe_2, wyniki_s_b_2, label='Susceptible (s)')
plt.plot(kroki_czasowe_2, wyniki_e_b_2, label='Exposed (e)')
plt.plot(kroki_czasowe_2, wyniki_i_b_2, label='Infectious (i)')
plt.plot(kroki_czasowe_2, wyniki_r_b_2, label='Removed (r)')
plt.title('SEIR Model - Metoda Rungego-Kutty czwartego rzędu (β = 0.5)')
plt.xlabel('Czas')
plt.legend()



'''Zmniejszenie parametru β o połowę oznacza redukcję liczby kontaktów, tym samym nowych zakażeń,
co powoduje spowolnienie tempa rozprzestrzeniania się epidemii. 
Na wykresie widać, że krzywa przebiegu epidemii jest bardziej płaska, 
a szczyt liczby zainfekowanych przypadków niższy, 
może jednak prowadzić to do wydłużenia czasu jej trwania.'''


# Wykresy dla zadania 3 
plt.subplot(3, 2, 3)
plt.plot(kroki_czasowe_wieksze_R0, wyniki_s_wieksze_R0, label='Susceptible (s)')
plt.plot(kroki_czasowe_wieksze_R0, wyniki_e_wieksze_R0, label='Exposed (e)')
plt.plot(kroki_czasowe_wieksze_R0, wyniki_i_wieksze_R0, label='Infectious (i)')
plt.plot(kroki_czasowe_wieksze_R0, wyniki_r_wieksze_R0, label='Removed (r)')
plt.title('SEIR Model - Metoda Rungego-Kutty czwartego rzędu (Większe R0)')
plt.xlabel('Czas')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(kroki_czasowe_mniejsze_R0, wyniki_s_mniejsze_R0, label='Susceptible (s)')
plt.plot(kroki_czasowe_mniejsze_R0, wyniki_e_mniejsze_R0, label='Exposed (e)')
plt.plot(kroki_czasowe_mniejsze_R0, wyniki_i_mniejsze_R0, label='Infectious (i)')
plt.plot(kroki_czasowe_mniejsze_R0, wyniki_r_mniejsze_R0, label='Removed (r)')
plt.title('SEIR Model - Metoda Rungego-Kutty czwartego rzędu (Mniejsze R0)')
plt.xlabel('Czas')
plt.legend()

plt.tight_layout()
plt.show()



'''Zgodnie z opisem przy wartosci R0>1 parametru reprodukcji liczba zakażonych rośnie  
- średnio jeden chory zaraża więcej niż jedną osobę, co prowadzi do wzrostu liczby zakażeń, 
a epidemia wciąż się rozwija. 
Jeśli parametr reprodukcji jest mniejszy od 1, epidemia jest kontrolowana, 
a liczba zakażeń maleje prowadząc do wygaśnięcia epidemii. Wykresy potwierdzają rolę R0 
w dynamice epidemii i modelach epidemiologicznych.'''
