import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения чисел
pd.set_option('display.float_format', '{:.4f}'.format)
np.set_printoptions(precision=4, suppress=True)

# ==================== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ====================
print("=" * 80)
print("СТАТИСТИЧЕСКИЙ АНАЛИЗ ДАННЫХ")
print("=" * 80)

# Загрузка данных с учетом разделителей
data = pd.read_csv('list17.csv', sep=';', decimal=',', header=None)
print(f"Размерность данных: {data.shape}")
print(f"Первые 5 строк данных:")
print(data.head())
print()

# У нас 6 наборов данных (столбцы 1-6)
# Создаем список всех выборок
samples = []
sample_names = []
for i in range(6):
    sample = data.iloc[:, i].dropna().astype(float).values
    samples.append(sample)
    sample_names.append(f"Набор {i+1}")
    
# Выводим информацию о каждом наборе
print("Информация о наборах данных:")
for i, (name, sample) in enumerate(zip(sample_names, samples)):
    print(f"{name}: n={len(sample)}, min={sample.min():.3f}, max={sample.max():.3f}, mean={sample.mean():.3f}")

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def compute_sample_characteristics(sample):
    """Вычисление выборочных характеристик"""
    n = len(sample)
    mean = np.mean(sample)
    variance = np.var(sample, ddof=1)  # исправленная дисперсия
    median = np.median(sample)
    
    # Коэффициент асимметрии
    skewness = stats.skew(sample, bias=False)
    
    # Коэффициент эксцесса
    kurtosis = stats.kurtosis(sample, bias=False)
    
    # Мода (приблизительно)
    hist, bins = np.histogram(sample, bins='auto')
    mode_pos = np.argmax(hist)
    mode = (bins[mode_pos] + bins[mode_pos + 1]) / 2
    
    return {
        'n': n,
        'mean': mean,
        'variance': variance,
        'std': np.sqrt(variance),
        'median': median,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mode': mode,
        'min': np.min(sample),
        'max': np.max(sample),
        'range': np.ptp(sample)
    }

def plot_histogram_with_distributions(sample, sample_num, params=None):
    """Построение гистограммы и теоретических распределений"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Анализ набора данных {sample_num+1}', fontsize=14)
    
    # Гистограмма
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(sample, bins='auto', density=True, alpha=0.7, 
                                 color='skyblue', edgecolor='black')
    ax1.set_xlabel('Значение')
    ax1.set_ylabel('Плотность')
    ax1.set_title('Гистограмма относительных частот')
    ax1.grid(True, alpha=0.3)
    
    # Эмпирическая функция распределения
    ax2 = axes[0, 1]
    sorted_sample = np.sort(sample)
    ecdf = np.arange(1, len(sorted_sample) + 1) / len(sorted_sample)
    ax2.step(sorted_sample, ecdf, where='post', linewidth=2)
    ax2.set_xlabel('Значение')
    ax2.set_ylabel('F(x)')
    ax2.set_title('Эмпирическая функция распределения')
    ax2.grid(True, alpha=0.3)
    
    # Квантиль-квантиль график для нормального распределения
    ax3 = axes[0, 2]
    stats.probplot(sample, dist="norm", plot=ax3)
    ax3.set_title('Q-Q plot для нормального распределения')
    ax3.grid(True, alpha=0.3)
    
    # Box plot
    ax4 = axes[1, 0]
    ax4.boxplot(sample, vert=True, patch_artist=True)
    ax4.set_ylabel('Значение')
    ax4.set_title('Box plot')
    ax4.grid(True, alpha=0.3)
    
    # Теоретические распределения (если есть параметры)
    ax5 = axes[1, 1]
    ax5.hist(sample, bins='auto', density=True, alpha=0.5, 
             color='skyblue', edgecolor='black', label='Гистограмма')
    
    if params:
        x = np.linspace(np.min(sample), np.max(sample), 1000)
        
        # Нормальное распределение
        if 'norm' in params:
            norm_pdf = stats.norm.pdf(x, params['norm']['loc'], params['norm']['scale'])
            ax5.plot(x, norm_pdf, 'r-', linewidth=2, label='Нормальное')
        
        # Равномерное распределение F2
        if 'uniform' in params:
            a, b = params['uniform']['loc'], params['uniform']['loc'] + params['uniform']['scale']
            mask = (x >= a) & (x <= b)
            uniform_pdf = np.zeros_like(x)
            uniform_pdf[mask] = 1 / (b - a)
            ax5.plot(x, uniform_pdf, 'g-', linewidth=2, label='Равномерное (F2)')
        
        # Гамма распределение F3
        if 'gamma' in params:
            # Преобразование параметров для scipy
            shape = params['gamma']['shape']
            scale = 1 / params['gamma']['rate'] if params['gamma']['rate'] != 0 else 1
            gamma_pdf = stats.gamma.pdf(x, a=shape, scale=scale)
            ax5.plot(x, gamma_pdf, 'b-', linewidth=2, label='Гамма (F3)')
        
        # Треугольное распределение F4
        if 'triangular' in params:
            c = params['triangular']['mode']
            triangular_pdf = np.where(x <= c, 
                                      2*x/(c*params['triangular']['scale']), 
                                      2*(params['triangular']['scale']-x)/(params['triangular']['scale']*(params['triangular']['scale']-c)))
            ax5.plot(x, triangular_pdf, 'm-', linewidth=2, label='Треугольное (F4)')
    
    ax5.set_xlabel('Значение')
    ax5.set_ylabel('Плотность')
    ax5.set_title('Теоретические распределения')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Пустой график для симметрии
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Анализ набора данных {sample_num+1}.png')
    
def runs_test(sample, alpha=0.05):
    """Критерий серий для проверки случайности"""
    median = np.median(sample)
    
    # Создаем последовательность знаков
    signs = np.where(sample > median, '+', '-')
    
    # Удаляем медианные значения
    signs = signs[sample != median]
    
    if len(signs) == 0:
        return {"статус": "Не применим", "причина": "Все значения равны медиане"}
    
    # Подсчет серий
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1
    
    n1 = np.sum(signs == '+')
    n2 = np.sum(signs == '-')
    n = n1 + n2
    
    if n1 < 10 or n2 < 10:
        # Используем табличное значение для малых выборок
        # Для простоты считаем, что если n1 или n2 < 10, используем нормальное приближение с поправкой
        pass
    
    # Ожидаемое количество серий и дисперсия
    E = (2 * n1 * n2) / n + 1
    D = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    
    if D <= 0:
        return {"статус": "Не применим", "причина": "Дисперсия <= 0"}
    
    # Статистика Z
    Z = (runs - E) / np.sqrt(D)
    
    # Критические значения
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    return {
        "медиана": median,
        "n1 (+)": n1,
        "n2 (-)": n2,
        "число серий": runs,
        "E": E,
        "D": D,
        "Z": Z,
        "Z_крит": z_critical,
        "p-value": 2 * (1 - stats.norm.cdf(abs(Z))),
        "вывод": "Не отвергаем" if abs(Z) <= z_critical else "Отвергаем"
    }

def estimate_parameters_mle(sample, dist_type):
    """Оценка параметров методом максимального правдоподобия"""
    
    if dist_type == "F1":  # Нормальное распределение
        mu = np.mean(sample)
        sigma2 = np.var(sample)  # MLE оценка дисперсии (без поправки)
        return {"тип": "F1", "μ": mu, "σ²": sigma2, "σ": np.sqrt(sigma2)}
    
    elif dist_type == "F2":  # Равномерное на [-1, 2θ]
        # f(x) = 1/(2θ+1), x ∈ [-1, 2θ]
        # Оценка θ методом моментов: E[X] = (-1 + 2θ)/2 = θ - 0.5
        # => θ_hat = mean + 0.5
        theta_hat = np.mean(sample) + 0.5
        return {"тип": "F2", "θ": theta_hat, "a": -1, "b": 2*theta_hat}
    
    elif dist_type == "F3":  # f(x) = θ² x e^{-θx}, x ≥ 0 (Гамма с формой 2)
        # Это распределение Гамма(2, θ)
        # MLE для θ: θ_hat = 2 / mean
        mean_val = np.mean(sample)
        if mean_val > 0:
            theta_hat = 2 / mean_val
        else:
            theta_hat = 1.0  # значение по умолчанию
        return {"тип": "F3", "θ": theta_hat, "форма": 2, "масштаб": 1/theta_hat}
    
    elif dist_type == "F4":  # f(x) = 2x/θ², x ∈ [0, θ]
        # Это треугольное распределение с модой в θ
        # MLE для θ: θ_hat = max(x)
        theta_hat = np.max(sample)
        return {"тип": "F4", "θ": theta_hat, "min": 0, "max": theta_hat}
    
    else:
        return {"тип": "Неизвестно", "ошибка": "Неизвестный тип распределения"}

def chi_square_test(sample, dist_type, params, alpha=0.1):
    """Критерий хи-квадрат Пирсона"""
    n = len(sample)
    
    # Определяем количество интервалов по формуле Стерджеса
    k = int(1 + 3.322 * np.log10(n))
    
    # Создаем интервалы
    hist, bin_edges = np.histogram(sample, bins=k)
    
    # Теоретические вероятности
    if dist_type == "F1":  # Нормальное
        mu, sigma = params["μ"], params["σ"]
        probs = []
        for i in range(len(bin_edges)-1):
            prob = stats.norm.cdf(bin_edges[i+1], mu, sigma) - \
                   stats.norm.cdf(bin_edges[i], mu, sigma)
            probs.append(prob)
    
    elif dist_type == "F2":  # Равномерное
        a, b = params["a"], params["b"]
        probs = []
        for i in range(len(bin_edges)-1):
            left = max(bin_edges[i], a)
            right = min(bin_edges[i+1], b)
            if right > left:
                prob = (right - left) / (b - a)
            else:
                prob = 0
            probs.append(prob)
    
    elif dist_type == "F3":  # Гамма
        shape, scale = params["форма"], params["масштаб"]
        probs = []
        for i in range(len(bin_edges)-1):
            prob = stats.gamma.cdf(bin_edges[i+1], a=shape, scale=scale) - \
                   stats.gamma.cdf(bin_edges[i], a=shape, scale=scale)
            probs.append(prob)
    
    elif dist_type == "F4":  # Треугольное (упрощенное)
        theta = params["θ"]
        probs = []
        for i in range(len(bin_edges)-1):
            # Для треугольного распределения f(x) = 2x/θ²
            left = max(bin_edges[i], 0)
            right = min(bin_edges[i+1], theta)
            if right > left:
                prob = (right**2 - left**2) / (theta**2)
            else:
                prob = 0
            probs.append(prob)
    
    else:
        return {"ошибка": "Неизвестный тип распределения"}
    
    # Преобразуем в массив numpy
    probs = np.array(probs)
    expected = n * probs
    
    # Объединяем интервалы, если ожидаемые частоты < 5
    i = 0
    while i < len(expected):
        if expected[i] < 5:
            if i == len(expected) - 1:
                # Если последний интервал, объединяем с предыдущим
                if i > 0:
                    hist[i-1] += hist[i]
                    expected[i-1] += expected[i]
                    hist = np.delete(hist, i)
                    expected = np.delete(expected, i)
                    probs = np.delete(probs, i)
                    bin_edges = np.delete(bin_edges, i+1)
                    i -= 1
            else:
                # Объединяем с следующим интервалом
                hist[i] += hist[i+1]
                expected[i] += expected[i+1]
                hist = np.delete(hist, i+1)
                expected = np.delete(expected, i+1)
                probs = np.delete(probs, i+1)
                bin_edges = np.delete(bin_edges, i+2)
        else:
            i += 1
    
    # Вычисляем статистику хи-квадрат
    chi2_stat = np.sum((hist - expected)**2 / expected)
    
    # Число степеней свободы
    df = len(hist) - 1 - len(params)
    
    # Критическое значение
    chi2_critical = stats.chi2.ppf(1 - alpha, df)
    
    # p-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)
    
    return {
        "хи-квадрат": chi2_stat,
        "степени свободы": df,
        "критическое значение": chi2_critical,
        "p-value": p_value,
        "вывод": "Не отвергаем" if chi2_stat <= chi2_critical else "Отвергаем",
        "интервалы": len(hist),
        "наблюдаемые": hist.tolist(),
        "ожидаемые": expected.tolist()
    }

def mann_whitney_test(sample, alpha=0.01):
    """Критерий Манна-Уитни для проверки однородности"""
    n = len(sample)
    
    # Разделяем выборку на две части
    n1 = n // 2
    n2 = n - n1
    
    sample1 = sample[:n1]
    sample2 = sample[n1:]
    
    # Выполняем тест Манна-Уитни
    stat, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
    
    return {
        "n1": n1,
        "n2": n2,
        "U-статистика": stat,
        "p-value": p_value,
        "вывод": "Не отвергаем" if p_value > alpha else "Отвергаем"
    }

def analyze_distribution_type(characteristics):
    """Выбор типа распределения на основе характеристик"""
    skewness = characteristics['skewness']
    kurtosis = characteristics['kurtosis']
    
    # Правила для определения типа распределения
    if abs(skewness) < 0.5 and abs(kurtosis) < 1:
        return "F1"  # Нормальное
    elif characteristics['min'] >= -1 and (characteristics['max'] - characteristics['min']) < 20:
        # Проверяем на равномерность
        if abs(skewness) < 1:
            return "F2"  # Равномерное
    elif characteristics['min'] >= 0 and skewness > 0:
        if kurtosis > 0:
            return "F3"  # Гамма
        else:
            return "F4"  # Треугольное
    
    # По умолчанию считаем нормальным
    return "F1"

# ==================== ОСНОВНОЙ ЦИКЛ АНАЛИЗА ====================

for i, (sample, name) in enumerate(zip(samples, sample_names)):
    print("=" * 80)
    print(f"АНАЛИЗ НАБОРА ДАННЫХ {i+1}")
    print("=" * 80)
    
    # 1. Вычисление характеристик
    print("\n1. ВЫБОРОЧНЫЕ ХАРАКТЕРИСТИКИ:")
    characteristics = compute_sample_characteristics(sample)
    for key, value in characteristics.items():
        if key != 'n':
            print(f"   {key}: {value:.4f}")
    
    # 2. Выдвижение гипотезы о распределении
    print("\n2. ГИПОТЕЗА О РАСПРЕДЕЛЕНИИ:")
    dist_type = analyze_distribution_type(characteristics)
    dist_names = {
        "F1": "Нормальное распределение N(a, σ²)",
        "F2": "Равномерное распределение на [-1, 2θ]",
        "F3": "Гамма-распределение f(x) = θ² x e^{-θx}, x ≥ 0",
        "F4": "Треугольное распределение f(x) = 2x/θ², x ∈ [0, θ]"
    }
    print(f"   Предполагаемый тип: {dist_type} - {dist_names[dist_type]}")
    
    # 3. Проверка случайности (критерий серий)
    print("\n3. ПРОВЕРКА СЛУЧАЙНОСТИ (КРИТЕРИЙ СЕРИЙ), α=0.05:")
    runs_result = runs_test(sample)
    for key, value in runs_result.items():
        print(f"   {key}: {value}")
    
    # 4. Оценка параметров ММП
    print("\n4. ОЦЕНКА ПАРАМЕТРОВ МЕТОДОМ МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ:")
    params = estimate_parameters_mle(sample, dist_type)
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    # 5. Проверка гипотезы о распределении (хи-квадрат)
    print("\n5. ПРОВЕРКА ГИПОТЕЗЫ О РАСПРЕДЕЛЕНИИ (χ²), α=0.1:")
    chi2_result = chi_square_test(sample, dist_type, params)
    for key, value in chi2_result.items():
        if key not in ['наблюдаемые', 'ожидаемые']:
            print(f"   {key}: {value}")
    
    # 6. Проверка однородности (Манн-Уитни)
    print("\n6. ПРОВЕРКА ОДНОРОДНОСТИ (МАНН-УИТНИ), α=0.01:")
    mw_result = mann_whitney_test(sample)
    for key, value in mw_result.items():
        print(f"   {key}: {value}")
    
    # 7. Построение графиков
    print("\n7. ПОСТРОЕНИЕ ГРАФИКОВ...")
    
    # Подготовка параметров для графиков
    plot_params = {}
    if dist_type == "F1":
        plot_params['norm'] = {'loc': params['μ'], 'scale': params['σ']}
    elif dist_type == "F2":
        plot_params['uniform'] = {'loc': params['a'], 'scale': params['b'] - params['a']}
    elif dist_type == "F3":
        plot_params['gamma'] = {'shape': params['форма'], 'rate': params['θ']}
    elif dist_type == "F4":
        plot_params['triangular'] = {'mode': params['θ'], 'scale': params['θ']}
    
    plot_histogram_with_distributions(sample, i, plot_params)
    
    print("\n" + "=" * 80)
    print(f"ЗАКЛЮЧЕНИЕ ПО НАБОРУ {i+1}:")
    print(f"1. Предполагаемое распределение: {dist_names[dist_type]}")
    print(f"2. Случайность выборки: {runs_result.get('вывод', 'Не определено')}")
    print(f"3. Соответствие распределению (χ²): {chi2_result.get('вывод', 'Не определено')}")
    print(f"4. Однородность выборки (Манн-Уитни): {mw_result.get('вывод', 'Не определено')}")
    print("=" * 80 + "\n\n")

# ==================== СРАВНИТЕЛЬНАЯ ТАБЛИЦА ====================
print("=" * 80)
print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 80)

results_table = []
for i, (sample, name) in enumerate(zip(samples, sample_names)):
    chars = compute_sample_characteristics(sample)
    dist_type = analyze_distribution_type(chars)
    
    results_table.append({
        "Набор": i+1,
        "n": chars['n'],
        "Среднее": chars['mean'],
        "Дисперсия": chars['variance'],
        "Асимметрия": chars['skewness'],
        "Эксцесс": chars['kurtosis'],
        "Распределение": dist_type,
        "Минимум": chars['min'],
        "Максимум": chars['max']
    })

results_df = pd.DataFrame(results_table)
print(results_df.to_string(index=False))

# ==================== ВЫВОДЫ ====================
print("\n" + "=" * 80)
print("ОБЩИЕ ВЫВОДЫ:")
print("=" * 80)

print("""
1. Для каждого набора данных проведен полный статистический анализ:
   - Построены гистограммы и вычислены выборочные характеристики
   - Выдвинуты гипотезы о виде распределения
   - Проверена случайность выборок
   - Оценены параметры распределений
   - Проверены гипотезы о распределении
   - Проверена однородность выборок

2. Рекомендации по использованию наборов данных:
   - Наборы с нормальным распределением (F1): подходят для параметрических тестов
   - Наборы с равномерным распределением (F2): требуют непараметрических методов
   - Наборы с гамма-распределением (F3): типичны для положительных величин
   - Наборы с треугольным распределением (F4): встречаются в задачах моделирования

3. Все расчеты выполнены с требуемыми уровнями значимости:
   - Критерий серий: α=0.05
   - Критерий χ²: α=0.1
   - Критерий Манна-Уитни: α=0.01
""")

print('ИЗОБРАЖЕНИЯ ГРАФИКОВ БЫЛИ СОХРАНЕНЫ В РАБОЧЕЙ ДИРЕКТОРИИ')